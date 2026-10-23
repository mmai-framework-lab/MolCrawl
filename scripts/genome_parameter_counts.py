"""How many parameters each genome model trains, with the embeddings separated.

The number a compute figure needs is the non-embedding count. Embeddings barely
enter the matrix multiplies, and a vocabulary big enough to matter would put the
two counts far apart -- molnl's 50,264 symbols hold 39M parameters, so by total
size its small looks 1.45x protein's while both cost the same to train. genome's
single-nucleotide vocabulary is 10 symbols, so the gap here comes almost
entirely from the position table; this prints both so the claim can be checked
rather than assumed.

The two architectures are counted from what they actually saved: nanoGPT keeps a
state dict in ckpt.pt, and the BERT side keeps a safetensors file whose header
carries every tensor's shape without reading the weights.

What counts as an embedding differs between them, and the difference is not a
convention:

  BERT      word, position and token-type tables are all inputs, and none of
            them is reused as an output weight. All three are subtracted.

  nanoGPT   wpe is the position table and is subtracted. wte is *not*: the model
            ties it to the output head, so those parameters do the work of the
            final layer and are part of the matrix multiplies. This follows
            model.py's own get_num_params(non_embedding=True).

A count is reported per architecture, not per run: every run of a series shares
the architecture, and a figure that differs between two runs of one series is a
defect rather than a number to average.
"""
import argparse
import glob
import json
import os
import struct


def safetensors_shapes(path):
    """{tensor name: shape} from the header, without reading the weights."""
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def bert_counts(checkpoint_dir):
    shapes = safetensors_shapes(os.path.join(checkpoint_dir, "model.safetensors"))
    total = sum(numel(s) for s in shapes.values())
    embedding = sum(numel(s) for name, s in shapes.items()
                    if ".embeddings.word_embeddings." in name
                    or ".embeddings.position_embeddings." in name
                    or ".embeddings.token_type_embeddings." in name)
    # The names of the tables actually subtracted, not every tensor whose name
    # contains "embeddings": the LayerNorm that sits in the embedding module is
    # not an embedding and is not subtracted, and listing it as though it were
    # misstates where the number came from.
    subtracted = {name: numel(s) for name, s in shapes.items()
                  if ".embeddings.word_embeddings." in name
                  or ".embeddings.position_embeddings." in name
                  or ".embeddings.token_type_embeddings." in name}
    return total, embedding, subtracted


def gpt2_counts(ckpt_path):
    import torch

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"]
    clean = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    total = sum(v.numel() for v in clean.values())
    # wpe only: wte is tied to the output head, so it is not an embedding for
    # this purpose. model.py's get_num_params makes the same cut.
    wpe = sum(v.numel() for k, v in clean.items() if k.endswith("transformer.wpe.weight"))
    wte = sum(v.numel() for k, v in clean.items() if k.endswith("transformer.wte.weight"))
    return total, wpe, wte, blob.get("config", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--bert-run", default="bert-small-mammal_centered-w1026")
    ap.add_argument("--gpt2-run", default="gpt2-small-mammal_centered-ep9")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []

    run = os.path.join(args.runs_root, args.bert_run)
    cks = sorted(glob.glob(f"{run}/checkpoint-*/model.safetensors"),
                 key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    if not cks:
        raise SystemExit(f"no safetensors under {run}")
    total, emb, names = bert_counts(os.path.dirname(cks[-1]))
    cfg = json.load(open(os.path.join(os.path.dirname(cks[-1]), "config.json")))
    print(f"  BERT  {args.bert_run}")
    print(f"    checkpoint            {os.path.basename(os.path.dirname(cks[-1]))}")
    print(f"    hidden / layers / heads {cfg.get('hidden_size')} / "
          f"{cfg.get('num_hidden_layers')} / {cfg.get('num_attention_heads')}")
    print(f"    vocab / max_position    {cfg.get('vocab_size')} / "
          f"{cfg.get('max_position_embeddings')}")
    print("    引いた表の内訳          " + ", ".join(
        f"{n.split('.embeddings.')[1].replace('.weight', '')} {v:,}"
        for n, v in sorted(names.items())))
    print(f"    総パラメータ            {total:>13,}  ({total / 1e6:.2f}M)")
    print(f"    うち埋め込み            {emb:>13,}  ({emb / 1e6:.2f}M)")
    print(f"    非埋め込み              {total - emb:>13,}  ({(total - emb) / 1e6:.2f}M)")

    rows.append({"arch": "bert", "run": args.bert_run, "total": total,
                 "embedding": emb, "non_embedding": total - emb})

    run = os.path.join(args.runs_root, args.gpt2_run, "ckpt.pt")
    if not os.path.exists(run):
        raise SystemExit(f"no ckpt.pt at {run}")
    total, wpe, wte, cfg = gpt2_counts(run)
    print(f"\n  GPT-2  {args.gpt2_run}")
    print(f"    n_layer / n_head / n_embd {cfg.get('n_layer')} / "
          f"{cfg.get('n_head')} / {cfg.get('n_embd')}")
    # The checkpoint's config records meta_vocab_size only when the config set
    # it; wte's own shape always says what the model was built with.
    print(f"    vocab / block_size        {wte // cfg.get('n_embd', 1) if wte else '?'} / "
          f"{cfg.get('block_size')}")
    print(f"    総パラメータ            {total:>13,}  ({total / 1e6:.2f}M)")
    print(f"    うち位置表 wpe          {wpe:>13,}  ({wpe / 1e6:.2f}M)  ← 引く")
    print(f"    うち語彙表 wte          {wte:>13,}  ({wte / 1e6:.2f}M)  ← 出力側と共有、引かない")
    print(f"    非埋め込み              {total - wpe:>13,}  ({(total - wpe) / 1e6:.2f}M)")
    rows.append({"arch": "gpt2", "run": args.gpt2_run, "total": total,
                 "embedding": wpe, "non_embedding": total - wpe})

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        import csv

        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
