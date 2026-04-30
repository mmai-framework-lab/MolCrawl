#!/usr/bin/env python3
"""
Environment variable check common module

Centralize checking of LEARNING_SOURCE_DIR environment variable
"""

import os
import sys


def check_learning_source_dir():
    """
    Check and retrieve LEARNING_SOURCE_DIR environment variable

        Returns:
    str: Value of LEARNING_SOURCE_DIR

        Note:
    If the environment variable is not set, output an error message and exit(1)
    """
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        print("❌ ERROR: LEARNING_SOURCE_DIR environment variable is required!", file=sys.stderr)
        print("", file=sys.stderr)
        print("Please set it before running:", file=sys.stderr)
        print("  export LEARNING_SOURCE_DIR='dir'", file=sys.stderr)
        print(f"  python {sys.argv[0] if sys.argv else 'script'}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Available learning directories:", file=sys.stderr)
        try:
            # Estimate project root
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            dirs = [d for d in os.listdir(project_root) if d.startswith("learning_")]
            for d in sorted(dirs):
                print(f"  - {d}", file=sys.stderr)
        except Exception:
            print("  (unable to list directories)", file=sys.stderr)
        sys.exit(1)

    return learning_source_dir
