#!/usr/bin/env python3
"""
モデルタイプの画像一覧を取得するスクリプト
Web API から呼び出される画像一覧取得用のユーティリティ
"""

import json
import sys
import os

# プロジェクトルートのsrcディレクトリをパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)

if __name__ == "__main__":
    try:
        from utils.image_manager import list_images_in_model_dir

        if len(sys.argv) < 2:
            print(json.dumps({"error": "Model type required"}), file=sys.stderr)
            sys.exit(1)

        model_type = sys.argv[1]

        # 画像一覧を取得
        images = list_images_in_model_dir(model_type)

        # JSON形式で出力
        print(json.dumps(images))

    except Exception as e:
        print(json.dumps({"error": f"Failed to get images: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
