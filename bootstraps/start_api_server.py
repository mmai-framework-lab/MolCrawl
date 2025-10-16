"""
APIサーバー起動スクリプト
"""

import sys
from pathlib import Path
import uvicorn

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import os
    
    # 環境変数にPYTHONPATHを追加
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(project_root) not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
    
    print("="*80)
    print("🚀 MolCrawl Experiment Management API Server")
    print("="*80)
    print(f"API URL: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Database: {project_root}/experiment_data/experiments.db")
    print(f"Project Root: {project_root}")
    print("="*80)
    
    uvicorn.run(
        "src.experiment_tracker.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
