"""
GPU環境とDDP設定の確認
"""

import torch
import os


def check_environment():
    print("=== GPU環境確認 ===")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    print(f"GPU数: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} (メモリ: {props.total_memory / 1e9:.1f}GB)")

    print("\n=== 環境変数確認 ===")
    ddp_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for var in ddp_vars:
        value = os.environ.get(var, "未設定")
        print(f"{var}: {value}")

    print("\n=== DDP判定 ===")
    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"DDP実行: {ddp}")

    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"ローカルランク: {local_rank}")
        if local_rank >= torch.cuda.device_count():
            print(f"⚠️ エラー: LOCAL_RANK({local_rank}) >= GPU数({torch.cuda.device_count()})")
            return False

    return True


if __name__ == "__main__":
    success = check_environment()
    if not success:
        print("\n❌ 環境設定に問題があります")
        exit(1)
    else:
        print("\n✅ 環境設定は正常です")
