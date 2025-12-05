#!/usr/bin/env python3
"""
paths.pyからLEARNING_SOURCE_DIRを取得してJSONで出力するスクリプト
環境変数LEARNING_SOURCE_DIRが必要です
"""

import json
import os
import sys

# プロジェクトルートのsrcディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

try:
    from config.paths import (
        ABSOLUTE_LEARNING_SOURCE_PATH,
        LEARNING_SOURCE_DIR,
        PROJECT_ROOT,
    )

    # 結果をJSON形式で出力
    result = {
        "learning_source_dir": LEARNING_SOURCE_DIR,
        "project_root": PROJECT_ROOT,
        "absolute_path": ABSOLUTE_LEARNING_SOURCE_PATH,
    }

    print(json.dumps(result))

except ImportError as e:
    print(json.dumps({"error": f"Cannot import paths: {e}"}), file=sys.stderr)
    sys.exit(1)
except SystemExit:
    # paths.pyでの環境変数チェックによる終了をキャッチ
    print(
        json.dumps({"error": "LEARNING_SOURCE_DIR environment variable is not set"}),
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
    sys.exit(1)
