#!/usr/bin/env python3
"""
指定されたモデルタイプとファイル名の画像パスを取得するスクリプト
Web API から呼び出される画像パス取得用のユーティリティ
"""

import json
import sys
import os

# プロジェクトルートのsrcディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)

if __name__ == "__main__":
    try:
        from utils.image_manager import get_image_path

        if len(sys.argv) < 3:
            print(json.dumps({"error": "Model type and filename required"}), file=sys.stderr)
            sys.exit(1)

        model_type = sys.argv[1]
        filename = sys.argv[2]

        # 画像パスを取得
        image_path = get_image_path(model_type, filename)

        # JSON形式で出力
        result = {"model_type": model_type, "filename": filename, "path": image_path}
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"Failed to get image path: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
