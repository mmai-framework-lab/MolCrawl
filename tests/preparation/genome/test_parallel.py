"""Regression checks for bounded discovery and process/output isolation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'molcrawl/preparation/genome/parallel.py'
spec = importlib.util.spec_from_file_location('refseq_launcher', RUNNER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_pilot_does_not_request_any_entries_after_limit(self):
        # Deliberately infinite: consuming the corpus, or pre-listing it, would fail.
        inspected = []

        class Entry:
            def __init__(self, number):
                self.name = f'GCF_{number}.fna.gz'
                self.path = '/input/' + self.name
            def is_dir(self, follow_symlinks=False):
                return False
            def is_file(self):
                return True
            def stat(self):
                return type('Stat', (), {'st_size': 100})()

        def entries():
            for i in range(33):
                inspected.append(i)
                if i == 32:
                    raise AssertionError('Scanned beyond the pilot limit')
                yield Entry(i)

        scan = mock.MagicMock()
        scan.__enter__.return_value = entries()
        with mock.patch.object(launcher.os, 'scandir', return_value=scan):
            result = launcher.discover_fastas(Path('/input'), 32)
        self.assertEqual(len(result), 32)
        self.assertEqual(inspected, list(range(32)))

    def test_subprocess_sharding_and_interrupted_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'fasta/GCF'
            source.mkdir(parents=True)
            for i in range(12):
                (source / f'GCF_{i}.fna').write_text('>x\n' + 'ACGT' * (i + 1))
            (source / 'skip_cds_from_genomic.fna').write_text('>x\nACGT')
            # Fixture only for the subprocess command; no real tokenization claimed.
            pkg = root / 'fixture'
            module = pkg / 'molcrawl/preparation/genome/preprocess_fasta.py'
            module.parent.mkdir(parents=True)
            module.write_text(
                'import argparse\nfrom pathlib import Path\n'
                'def build_parser():\n'
                '    p=argparse.ArgumentParser()\n'
                '    for name in ("input","output","model","chunk_size","n_split_threshold","max_n_frac"):\n'
                '        p.add_argument("--"+name)\n'
                '    return p\n'
                'if __name__ == "__main__":\n'
                '    a=build_parser().parse_args();out=Path(a.output);out.mkdir(parents=True)\n'
                '    (out/"files.txt").write_text("\\n".join(sorted(p.name for p in Path(a.input).iterdir())))\n')
            for parent in module.parents:
                if parent == pkg:
                    break
                (parent / '__init__.py').touch()
            env = dict(os.environ, PYTHONPATH=str(pkg))
            # Supply the actual relocated launcher in an isolated test package.
            canonical = pkg / 'molcrawl/preparation/genome/parallel.py'
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(RUNNER.read_text())
            (canonical.parent / '__init__.py').touch()
            (canonical.parent.parent / '__init__.py').touch()
            output = root / 'output'
            command = [sys.executable, str(RUNNER), '--input', str(root/'fasta'),
                       '--output', str(output), '--workers', '2', '--shards', '4']
            subprocess.run(command, env=env, cwd=pkg, check=True, capture_output=True)
            names = [name for f in output.glob('shard_*/processed/files.txt')
                     for name in f.read_text().splitlines()]
            self.assertEqual(len(names), 12)
            self.assertEqual(len(set(names)), 12)
            subprocess.run(command, env=env, cwd=pkg, check=True, capture_output=True)
            self.assertEqual(json.loads((output/'summary.json').read_text())['failures'], [])
            (output/'shard_000/complete.json').unlink()
            (output/'shard_000/started.json').write_text('{}')
            failed = subprocess.run(command, env=env, cwd=pkg, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn('interrupted output', failed.stderr)
            pilot = root/'pilot'
            subprocess.run([sys.executable, str(RUNNER), '--input', str(root/'fasta'),
                            '--output', str(pilot), '--workers', '1', '--pilot', '3'],
                           env=env, cwd=pkg, check=True, capture_output=True)
            summary = json.loads((pilot/'summary.json').read_text())
            self.assertEqual(summary['selected_count'], 3)
            self.assertIsNone(summary['total_count'])
            self.assertIsNone(summary['total_input_bytes'])
            self.assertIn('scan_seconds', summary)

    def test_missing_entrypoint_fails_before_scanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'molcrawl'
            pkg.mkdir()
            (pkg/'__init__.py').touch()
            result = subprocess.run([sys.executable, str(RUNNER), '--input', '/not-scanned',
                                     '--output', '/not-created'],
                                    env=dict(os.environ, PYTHONPATH=tmp), cwd=tmp,
                                    capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Preprocessor startup failed', result.stderr)
        self.assertNotIn('Scanning', result.stdout)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'pyarrow required for Parquet integration')
    def test_real_pipeline_and_legacy_module(self):
        import gzip
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            for relative in (
                'molcrawl/preparation/genome/__init__.py',
                'molcrawl/preparation/genome/parallel.py',
                'molcrawl/preparation/genome/preprocess_fasta.py',
                'molcrawl/data/genome_sequence/dataset/refseq/preprocess_fasta.py',
            ):
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
                for parent in target.parents:
                    if parent == repo:
                        break
                    (parent / '__init__.py').touch(exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(repo))
            source = root / 'input/GCF'
            source.mkdir(parents=True)
            sequences = {
                'GCF_shortN': 'A' * 1023 + 'N' + 'C' * 1024,
                'GCF_gap': 'A' * 1024 + 'N' * 10 + 'C' * 1024,
                'GCF_ambiguous': 'R' * 1024 + 'G' * 1024,
            }
            for name, seq in sequences.items():
                with gzip.open(source / (name + '.fna.gz'), 'wt') as out:
                    out.write('>test\n' + seq + '\n')
            canonical = 'molcrawl.preparation.genome.preprocess_fasta'
            legacy = 'molcrawl.data.genome_sequence.dataset.refseq.preprocess_fasta'
            outputs = []
            for index, module in enumerate((canonical, legacy)):
                output = root / f'direct_{index}'
                subprocess.run([sys.executable, '-m', module, '--input', str(source),
                                '--output', str(output), '--model', 'gpt2'],
                               env=env, cwd=repo, check=True, capture_output=True)
                outputs.append({p.stem: pq.read_table(p).to_pydict()
                                for p in (output / 'parquet_gpt2').glob('*.parquet')})
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[0]['GCF_shortN']['input_ids'],
                             [[0] * 1023 + [4], [3] * 1024])
            self.assertEqual(outputs[0]['GCF_gap']['input_ids'], [[0] * 1024, [3] * 1024])
            self.assertEqual(outputs[0]['GCF_ambiguous']['input_ids'], [[2] * 1024])
            named_import = subprocess.run([
                sys.executable, '-c',
                f'from {legacy} import normalize_sequence; print(normalize_sequence("aR"))'],
                env=env, cwd=repo, check=True, capture_output=True, text=True)
            self.assertEqual(named_import.stdout.strip(), 'AN')
            parallel_out = root / 'parallel'
            cmd = [sys.executable, '-m', 'molcrawl.preparation.genome.parallel',
                   '--input', str(root/'input'), '--output', str(parallel_out),
                   '--workers', '2', '--shards', '2', '--pilot', '3']
            subprocess.run(cmd, env=env, cwd=repo, check=True, capture_output=True)
            parallel_tables = {p.stem: pq.read_table(p).to_pydict()
                               for p in parallel_out.glob('shard_*/processed/parquet_gpt2/*.parquet')}
            self.assertEqual(parallel_tables, outputs[0])
            complete = parallel_out / 'shard_000/complete.json'
            before = complete.stat().st_mtime_ns
            plan = json.loads((parallel_out/'plan.json').read_text())
            self.assertEqual(plan['version'], 4)
            self.assertTrue(plan['pipeline'].startswith(legacy + ':'))
            subprocess.run(cmd, env=env, cwd=repo, check=True, capture_output=True)
            self.assertEqual(complete.stat().st_mtime_ns, before)

    def test_slurm_paths_and_arguments_are_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'checkout'
            module = repo / 'molcrawl/preparation/genome/parallel.py'
            module.parent.mkdir(parents=True)
            module.touch()
            calls = root / 'calls.jsonl'
            fake_python = root / 'python'
            fake_python.write_text(
                f'#!{sys.executable}\nimport json,sys\n'
                f'with open({str(calls)!r},"a") as out: out.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake_python.chmod(0o755)
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(repo),
                       MOLCRAWL_PROJECT_DIR=str(repo), MOLCRAWL_PYTHON=str(fake_python),
                       REFSEQ_BASE=str(root/'data'), REFSEQ_WORKERS='4')
            subprocess.run(['bash', str(ROOT/'workflows/01-genome_sequence-prepare-refseq.slurm'), 'full'],
                           env=env, check=True, capture_output=True)
            arguments = json.loads(calls.read_text().splitlines()[-1])
            self.assertEqual(arguments[:3], ['-u', '-m', 'molcrawl.preparation.genome.parallel'])
            self.assertEqual(arguments[arguments.index('--workers')+1], '4')
            self.assertEqual(arguments[arguments.index('--output')+1],
                             str(root/'data/processed_gpt2_n10_parallel'))


if __name__ == '__main__':
    unittest.main()
