"""Stage the rna 2026-11-18 extension: copy each run's resume point into its own tree.

rna-judgment-2026-11-18b asks for the extension to leave the nine-epoch runs intact.
Neither framework does that by itself:

  GPT-2  ``init_from="resume"`` reads ``out_dir/ckpt.pt`` and rewrites it at the first
         improvement, and ``own_checkpoint_steps`` counts everything at or below the
         resumed iter as this run's own, so cleanup may delete it. nanoGPT saves only
         on improvement, so that single file is the only copy of the nine-epoch
         weights -- medium's are at iter 100,100, not 120,960.

  BERT   ``BestNCheckpointRetention`` keeps the best ten by ``eval_loss_mask`` plus
         the newest, and its ``_own_steps`` reads ``state.log_history``, which HF
         restores on resume. So a resumed run owns the earlier checkpoints and can
         prune them -- including the one each arm's reported nine-epoch number rests
         on.

Both are answered the same way: give the extension its own output directory and put a
copy of the resume point in it. The extension configs already name those directories;
this script asks them rather than being told, so a path appears in one place only.

The scheduler is the other half, for BERT. HF restores it from the checkpoint and
``LambdaLR.load_state_dict`` does ``self.__dict__.update(state_dict)``, which brings
back ``base_lrs`` and ``last_epoch``. Read off a real rna checkpoint-120960:
``base_lrs [2.5e-05, 2.5e-05]``, ``last_epoch 120960``, ``_last_lr [0.0, 0.0]``. An
extension that inherits that runs at the ORIGINAL peak with the warmup already behind
it, so the config's ``learning_rate`` and ``warmup_steps`` are both discarded.

Two ways out, and ``--scheduler`` picks one rather than assuming:

  remove   delete scheduler.pt from the copy. What workflows/bert-grid.sbatch
           instructs. In transformers 4.45.1 one ``if`` guards both loads, so this
           may also skip optimizer.pt -- see scripts/probe_extend_resume_scheduler.py.
  rewrite  keep the file and re-point it: base_lrs to this config's rate, last_epoch
           to 0. The optimizer still loads because the file is still there.

Nothing here is launched. The script copies, reports, and stops.
"""

import argparse
import importlib
import os
import shutil
import sys

CONFIG_PKG = "molcrawl.tasks.pretrain.configs.rna"
RESUME_STEP = 120960

# (framework, source arm config, extension config)
PAIRS = [
    ("bert", "bert_small_lr3e4", "bert_small_lr3e4_ext242k"),
    ("bert", "bert_medium_lr3e4", "bert_medium_lr3e4_ext242k"),
    ("bert", "bert_large_lr1p7e4", "bert_large_lr1p7e4_ext242k"),
    ("bert", "bert_xl_lr2p5e5", "bert_xl_lr2p5e5_ext242k"),
    ("gpt2", "gpt2_small_lr1p2e3_9ep", "gpt2_small_lr1p2e3_9ep_ext242k"),
    ("gpt2", "gpt2_medium_lr1p2e3_9ep", "gpt2_medium_lr1p2e3_9ep_ext242k"),
    ("gpt2", "gpt2_large_lr6e4_9ep", "gpt2_large_lr6e4_9ep_ext242k"),
    ("gpt2", "gpt2_xl_lr6e4_9ep", "gpt2_xl_lr6e4_9ep_ext242k"),
]


def load(name):
    return importlib.import_module(f"{CONFIG_PKG}.{name}")


def run_dir(framework, mod):
    """Where a config writes. The config is the only place the path is spelled."""
    key = "model_path" if framework == "bert" else "out_dir"
    path = getattr(mod, key, None)
    if not path:
        raise SystemExit(f"{mod.__name__} sets no {key}")
    return path


def tree_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0


def rewrite_scheduler(path, new_lr):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    before = {
        "base_lrs": [float(x) for x in state["base_lrs"]],
        "last_epoch": state["last_epoch"],
    }
    state["base_lrs"] = [float(new_lr)] * len(state["base_lrs"])
    state["last_epoch"] = 0
    state["_step_count"] = 1
    state["_last_lr"] = [0.0] * len(state["base_lrs"])
    if "_is_initial" in state:
        state["_is_initial"] = True
    torch.save(state, path)
    return before


def gpt2_iter(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("iter_num"), ckpt.get("best_val_loss")


def stage_bert(src, dst, ext_mod, scheduler_mode, dry, verify):
    src_ckpt = os.path.join(src, f"checkpoint-{RESUME_STEP}")
    dst_ckpt = os.path.join(dst, f"checkpoint-{RESUME_STEP}")
    if not os.path.isdir(src_ckpt):
        raise SystemExit(f"no {src_ckpt} to extend from")
    size = tree_bytes(src_ckpt)
    print(f"    from   {src_ckpt}")
    print(f"    to     {dst_ckpt}   ({human(size)})")
    print(f"    scheduler.pt: {scheduler_mode}")
    if dry:
        return size
    if os.path.exists(dst_ckpt):
        raise SystemExit(f"{dst_ckpt} already exists -- refusing to overwrite a staged copy")
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src_ckpt, dst_ckpt)
    sched = os.path.join(dst_ckpt, "scheduler.pt")
    if scheduler_mode == "remove":
        if os.path.exists(sched):
            os.remove(sched)
            print("    removed scheduler.pt from the copy")
    else:
        before = rewrite_scheduler(sched, ext_mod.learning_rate)
        print(f"    rewrote scheduler.pt: base_lrs {before['base_lrs']} -> "
              f"{ext_mod.learning_rate}, last_epoch "
              f"{before['last_epoch']} -> 0")
    if verify:
        import json as _json

        ts = os.path.join(dst_ckpt, "trainer_state.json")
        st = _json.load(open(ts, encoding="utf-8"))
        print(f"    copy holds global_step {st.get('global_step')} "
              f"max_steps {st.get('max_steps')}, optimizer.pt "
              f"{'present' if os.path.exists(os.path.join(dst_ckpt, 'optimizer.pt')) else 'MISSING'}, "
              f"scheduler.pt {'present' if os.path.exists(sched) else 'absent'}")
    return size


def stage_gpt2(src, dst, dry, verify):
    src_ckpt = os.path.join(src, "ckpt.pt")
    dst_ckpt = os.path.join(dst, "ckpt.pt")
    if not os.path.isfile(src_ckpt):
        raise SystemExit(f"no {src_ckpt} to extend from")
    size = os.path.getsize(src_ckpt)
    print(f"    from   {src_ckpt}")
    print(f"    to     {dst_ckpt}   ({human(size)})")
    if dry:
        return size
    if os.path.exists(dst_ckpt):
        raise SystemExit(f"{dst_ckpt} already exists -- refusing to overwrite a staged copy")
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)
    if verify:
        it, best = gpt2_iter(dst_ckpt)
        print(f"    copy holds iter_num {it}, best_val_loss {best}")
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", choices=("remove", "rewrite"), required=True,
                    help="how the BERT copies hand scheduler.pt over; see the module docstring")
    ap.add_argument("--only", default="", help="substring filter on the extension config name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="read each copy back after writing it")
    a = ap.parse_args()

    if not os.environ.get("MODEL_OUTPUT_ROOT"):
        raise SystemExit("set MODEL_OUTPUT_ROOT; the configs derive every path from it")

    total = 0
    for framework, arm, ext in PAIRS:
        if a.only and a.only not in ext:
            continue
        src_mod, ext_mod = load(arm), load(ext)
        src, dst = run_dir(framework, src_mod), run_dir(framework, ext_mod)
        print(f"\n== {ext}  ({framework})")
        if src == dst:
            raise SystemExit(f"{ext} writes where {arm} did ({src}) -- the point is to separate them")
        if framework == "bert":
            total += stage_bert(src, dst, ext_mod, a.scheduler, a.dry_run, a.verify)
        else:
            total += stage_gpt2(src, dst, a.dry_run, a.verify)

    print(f"\n=== {'would copy' if a.dry_run else 'copied'} {human(total)} in total")
    root = os.environ["MODEL_OUTPUT_ROOT"]
    st = os.statvfs(root)
    print(f"=== free under {root}: {human(st.f_bavail * st.f_frsize)}")
    if a.dry_run:
        print("=== dry run: nothing was written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
