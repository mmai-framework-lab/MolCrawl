#!/usr/bin/env python3
"""Partition FASTAs and run the existing MolCrawl CLI in separate processes."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import importlib

MODULE = 'molcrawl.preparation.genome.preprocess_fasta'
# Keep the plan identifier stable: relocation does not change preprocessing.
# This also lets already completed shards from run_parallel.py be reused.
PIPELINE = ('molcrawl.data.genome_sequence.dataset.refseq.preprocess_fasta'
            ':gpt2:n_split=10:max_n_frac=0.1:chunk=1024')


def check_molcrawl():
    module = importlib.import_module(MODULE)
    # Validate both the module's executable entry point and the required options.
    result = subprocess.run([sys.executable, '-m', MODULE, '--help'],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    module.build_parser().parse_args([
        '--input', '/preflight', '--output', '/preflight-output', '--model', 'gpt2',
        '--chunk_size', '1024', '--n_split_threshold', '10', '--max_n_frac', '0.1'])
    print(f'MolCrawl module OK: {module.__file__}', flush=True)
    print('Settings: GPT-2; 1024 nt; N-run split >=10; max N fraction 0.1', flush=True)


def discover_fastas(source, limit=0):
    """Walk one directory entry at a time; stop immediately after limit files."""
    suffixes = ('.fna.gz', '.fa.gz', '.fasta.gz', '.fna', '.fa', '.fasta')
    progress = dict(directories=0, entries=0, files=0, directory=str(source))
    stopped = threading.Event()

    def report():
        while not stopped.wait(15):
            print(f"Scan progress: directories={progress['directories']:,} "
                  f"entries={progress['entries']:,} FASTAs={progress['files']:,}; "
                  f"directory={progress['directory']}", flush=True)

    def walk(directory):
        progress['directory'] = str(directory)
        progress['directories'] += 1
        with os.scandir(directory) as entries:
            for entry in entries:
                progress['entries'] += 1
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(entry.path)
                elif entry.name.lower().endswith(suffixes):
                    lower = entry.name.lower()
                    if '_cds_from_genomic' in lower or '_rna_from_genomic' in lower:
                        continue
                    if entry.is_file():
                        yield entry.path, entry.stat().st_size

    heartbeat = threading.Thread(target=report, daemon=True)
    heartbeat.start()
    files = []
    stems = set()
    iterator = walk(source)
    try:
        for path, size in iterator:
            stem = re.sub(r'\.(fa|fna|fasta)(\.gz)?$', '', Path(path).name, flags=re.I)
            if stem in stems:
                raise RuntimeError(f'Duplicate FASTA stem: {stem}; resolve before running.')
            stems.add(stem)
            files.append((path, size))
            progress['files'] = len(files)
            if limit and len(files) >= limit:
                break
    finally:
        iterator.close()
        stopped.set()
        heartbeat.join()
    print(f"Scan finished: {len(files):,} FASTAs; "
          f"{progress['directories']:,} directories; {progress['entries']:,} entries."
          + (' Pilot limit reached; remaining input was NOT scanned.'
             if limit and len(files) >= limit else ''), flush=True)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shards', type=int, default=64)
    parser.add_argument('--pilot', type=int, default=0,
                        help='Stop discovery after the first N FASTAs; 0 uses all files.')
    args = parser.parse_args()
    if args.workers < 1 or args.shards < 1 or args.pilot < 0:
        parser.error('workers/shards must be positive; pilot must be nonnegative')
    # Check the actual executable entry point BEFORE touching the large input tree.
    check = subprocess.run([sys.executable, '-m', 'molcrawl.preparation.genome.parallel', '--check-molcrawl'],
                           capture_output=True, text=True)
    if check.returncode:
        raise SystemExit('Preprocessor startup failed:\n' + check.stdout + check.stderr)
    print(check.stdout, end='', flush=True)
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        parser.error(f'Input not found: {source}')
    if output == source or source in output.parents:
        parser.error('Output must be outside the input tree')
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / '.launcher.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = output / 'plan.json'
    started = time.monotonic()

    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get('version') != 4:
            raise SystemExit('Old input plan. Use a new output directory for the N>=10 pipeline.')
        if plan.get('pipeline') != PIPELINE:
            raise SystemExit('Existing plan uses a different MolCrawl pipeline. Use a new output directory.')
        if (plan['input'], plan['shards_requested'], plan['pilot']) != (
                str(source), args.shards, args.pilot):
            raise SystemExit('Existing plan differs. Choose a new output directory.')
        print('Reusing saved input plan (input tree must be unchanged).', flush=True)
    else:
        print('RefSeq launcher N>=10 pipeline (version 4)', flush=True)
        print(f'Scanning {source}; '
              + (f'stop after {args.pilot} FASTAs (no full scan).' if args.pilot
                 else 'full scan with progress every 15 seconds.'), flush=True)
        files = discover_fastas(source, limit=args.pilot)
        files.sort()
        if not files:
            raise SystemExit('No FASTA files found.')
        # A bounded pilot intentionally does not discover corpus totals.
        total_count = None if args.pilot else len(files)
        total_bytes = None if args.pilot else sum(size for _, size in files)
        n_shards = min(args.shards, len(files))
        groups = [[] for _ in range(n_shards)]
        loads = [0] * n_shards
        for item in sorted(files, key=lambda item: item[1], reverse=True):
            index = min(range(n_shards), key=lambda j: loads[j])
            groups[index].append(item)
            loads[index] += item[1]
        plan = dict(version=4, pipeline=PIPELINE, input=str(source), shards_requested=args.shards, pilot=args.pilot,
                    total_count=total_count, total_bytes=total_bytes,
                    selected_count=len(files), selected_bytes=sum(loads), groups=groups)
        tmp = plan_path.with_suffix('.json.part')
        with open(tmp, 'w') as out:
            json.dump(plan, out)
        os.replace(tmp, plan_path)

    scan_seconds = time.monotonic() - started
    print(f"Selected {plan['selected_count']:,} FASTAs; "
          f"{plan['selected_bytes'] / 2**30:.2f} GiB input; "
          f"{args.workers} concurrent processes", flush=True)
    if args.pilot:
        print('Quick pilot uses the first files found, not a representative sample. '
              'It does not test maximum assembly size or predict full runtime.', flush=True)
    processing_started = time.monotonic()

    def run_shard(item):
        index, group = item
        root = output / f'shard_{index:03d}'
        stage = root / 'input_links'
        result = root / 'processed'
        complete = root / 'complete.json'
        running = root / 'started.json'
        root.mkdir(exist_ok=True)
        if complete.exists():
            print(f'shard {index}: complete, skipped', flush=True)
            return json.loads(complete.read_text())
        if running.exists() or result.exists():
            raise RuntimeError(f'{root}: interrupted output. Move this entire shard directory '
                               'aside before retrying; the original CLI cannot safely resume partial files.')
        stage.mkdir(exist_ok=True)
        for name, _size in group:
            src = Path(name)
            link = stage / src.name
            if link.is_symlink():
                if link.resolve() != src.resolve():
                    raise RuntimeError(f'Unexpected link: {link}')
            elif link.exists():
                raise RuntimeError(f'Unexpected input file: {link}')
            else:
                link.symlink_to(src)
        running.write_text(json.dumps(dict(started=time.time())))
        command = [sys.executable, '-u', '-m', MODULE,
                   '--input', str(stage), '--output', str(result), '--model', 'gpt2',
                   '--chunk_size', '1024', '--n_split_threshold', '10', '--max_n_frac', '0.1']
        print(f'shard {index}: starting {len(group)} files', flush=True)
        t0 = time.monotonic()
        with open(root / 'preprocess.log', 'w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            tail = (root / 'preprocess.log').read_text(errors='replace').splitlines()[-25:]
            raise RuntimeError(f'Preprocessing exited {completed.returncode}. '
                               f'Log: {root / "preprocess.log"}\n' + '\n'.join(tail))
        stats = dict(shard=index, files=len(group), input_bytes=sum(size for _, size in group),
                     seconds=time.monotonic() - t0)
        complete.write_text(json.dumps(stats, indent=2))
        running.unlink()
        print(f"shard {index}: done in {stats['seconds']:.1f}s", flush=True)
        return stats

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_shard, item): item[0]
                   for item in enumerate(plan['groups'])}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
                print(f'FAILED shard {futures[future]}: {error}', file=sys.stderr, flush=True)
    summary = dict(total_count=plan['total_count'], total_input_bytes=plan['total_bytes'],
                   selected_count=plan['selected_count'], selected_input_bytes=plan['selected_bytes'],
                   workers=args.workers, elapsed_seconds=time.monotonic() - started,
                   scan_seconds=scan_seconds,
                   preprocessing_seconds=time.monotonic() - processing_started,
                   failures=failures,
                   note='Pilot uses first N files and does not scan the full tree. '
                        'It is not representative and cannot establish maximum RAM or full runtime. '
                        'Elapsed time on a resumed run includes skipped shards.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-molcrawl']:
        check_molcrawl()
    else:
        main()
