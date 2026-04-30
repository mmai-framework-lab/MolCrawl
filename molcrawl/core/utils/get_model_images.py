#!/usr/bin/env python3
"""
Script to get a list of model type images
Utility for obtaining a list of images called from Web API
"""

import json
import sys
import os

# Add project root src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)

if __name__ == "__main__":
    try:
        from molcrawl.core.utils.image_manager import list_images_in_model_dir

        if len(sys.argv) < 2:
            print(json.dumps({"error": "Model type required"}), file=sys.stderr)
            sys.exit(1)

        model_type = sys.argv[1]

        # Get image list
        images = list_images_in_model_dir(model_type)

        # Output in JSON format
        print(json.dumps(images))

    except Exception as e:
        print(json.dumps({"error": f"Failed to get images: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
