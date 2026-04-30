#!/usr/bin/env python3
"""
Script to get image path for specified model type and file name
Utility for obtaining image path called from Web API
"""

import json
import sys
import os

# Add project root src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)

if __name__ == "__main__":
    try:
        from molcrawl.core.utils.image_manager import get_image_path

        if len(sys.argv) < 3:
            print(json.dumps({"error": "Model type and filename required"}), file=sys.stderr)
            sys.exit(1)

        model_type = sys.argv[1]
        filename = sys.argv[2]

        # get image path
        image_path = get_image_path(model_type, filename)

        # Output in JSON format
        result = {"model_type": model_type, "filename": filename, "path": image_path}
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"Failed to get image path: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
