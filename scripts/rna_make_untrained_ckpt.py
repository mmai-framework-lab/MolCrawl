"""学習していない同じ形のモデルを、checkpoint の形で書き出す。

プローブの床を測るためのもの。床とは、細胞種の判別のうち、事前学習ではなく
形と語彙とトークン化から来ている分である。

評価器に「未学習で」という引数を足すのではなく、未学習の重みを checkpoint に
して既存の経路へ渡す。評価の経路が学習済みの 13 点とビット単位で同じになり、
床と本番の差が経路の違いでないと言えるようになる。評価器は 5 モダリティの
共有物なので、そこへ分岐を足さずに済む利点もある。

形は学習済みの checkpoint の model_args から取る。手で書くと、層数や幅が
1 つずれただけで床が別物になり、それが差として読まれる。
"""
import argparse
import os

import torch


def strip_compile_prefix(state):
    pre = "_orig_mod."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in state.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-checkpoint", required=True,
                    help="形を取ってくる学習済み checkpoint。重みは使わない")
    ap.add_argument("--out", required=True, help="書き出す checkpoint のパス")
    ap.add_argument("--seed", type=int, required=True, help="初期化の種")
    a = ap.parse_args(argv)

    from molcrawl.models.gpt2.model import GPT, GPTConfig

    ck = torch.load(a.from_checkpoint, map_location="cpu", weights_only=False)
    margs = dict(ck["model_args"])
    print(f"  形: {margs}")

    torch.manual_seed(a.seed)
    model = GPT(GPTConfig(**margs))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  seed {a.seed} で初期化  パラメータ {n_params/1e6:.2f}M")

    # 学習済みと重みが違うことを確かめる。同じ形なので、取り違えると
    # 床のつもりで本番を測ってしまい、しかも数字は自然に見える。
    trained = strip_compile_prefix(ck["model"])
    fresh = model.state_dict()
    key = next(k for k in fresh if fresh[k].dim() > 1)
    if torch.equal(fresh[key], trained[key].to(fresh[key].dtype)):
        raise SystemExit(f"{key} が学習済みと一致した。初期化されていない")
    print(f"  {key} は学習済みと異なる")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"model": fresh, "model_args": margs, "iter_num": 0,
                "best_val_loss": None, "untrained_seed": a.seed}, a.out)
    print(f"  WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
