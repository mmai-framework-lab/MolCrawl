"""
実験管理システムの基本テスト
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent


def test_imports():
    """モジュールのインポートテスト"""
    print("Testing imports...")
    try:
        from experiment_tracker import (  # noqa: F401
            DatasetType,
            ExperimentStatus,
            ExperimentTracker,
            ExperimentType,
            ModelType,
        )

        print("  ✓ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_database():
    """データベース接続テスト"""
    print("\nTesting database...")
    try:
        import os
        import tempfile

        from experiment_tracker.database import ExperimentDatabase

        # 一時ファイルを使用
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        db = ExperimentDatabase(db_path)
        stats = db.get_statistics()
        assert stats["total_experiments"] == 0

        # クリーンアップ
        import shutil

        shutil.rmtree(temp_dir)

        print("  ✓ Database connection successful")
        return True
    except Exception as e:
        import traceback

        print(f"  ✗ Database test failed: {e}")
        print(traceback.format_exc())
        return False


def test_tracker():
    """トラッカーの基本機能テスト"""
    print("\nTesting tracker...")
    try:
        import os

        # テスト用の一時データベース
        import tempfile

        from experiment_tracker import (
            DatasetType,
            ExperimentTracker,
            ExperimentType,
            ModelType,
        )

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        tracker = ExperimentTracker(db_path)

        # 実験作成
        exp_id = tracker.start_experiment(
            name="Test Experiment",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE,
        )
        print(f"    Created experiment: {exp_id}")

        # ステップ追加
        tracker.start_step(exp_id, "test_step", "Test Step")
        time.sleep(0.1)
        tracker.complete_step(exp_id, "test_step")
        print("    Added and completed step")

        # ログ追加
        tracker.log(exp_id, "INFO", "Test log message")
        print("    Added log entry")

        # 実験完了
        tracker.complete_experiment(exp_id, metrics={"accuracy": 0.95})
        print("    Completed experiment")

        # 取得テスト
        experiment = tracker.get_experiment(exp_id)
        assert experiment is not None
        assert len(experiment.steps) == 1
        assert len(experiment.logs) >= 2  # start + test log
        assert experiment.metrics["accuracy"] == 0.95
        print("    Retrieved and verified experiment")

        # クリーンアップ
        import shutil

        shutil.rmtree(temp_dir)

        print("  ✓ Tracker test successful")
        return True
    except Exception as e:
        import traceback

        print(f"  ✗ Tracker test failed: {e}")
        print(traceback.format_exc())
        return False


def test_helpers():
    """ヘルパー関数テスト"""
    print("\nTesting helpers...")
    try:
        import tempfile

        from experiment_tracker import (
            DatasetType,
            ExperimentType,
            ModelType,
        )
        from experiment_tracker.helpers import experiment_context

        temp_dir = tempfile.mkdtemp()

        # コンテキストマネージャーテスト
        with experiment_context(
            name="Test Context Experiment",
            experiment_type=ExperimentType.EVALUATION,
            model_type=ModelType.BERT,
            dataset_type=DatasetType.CLINVAR,
        ) as exp:
            exp.log("INFO", "Test log")
            exp.start_step("step1", "Test Step 1")
            exp.complete_step("step1")
            exp.add_metric("test_metric", 0.88)

        print("    Context manager test passed")

        # クリーンアップ
        import shutil

        shutil.rmtree(temp_dir)

        print("  ✓ Helpers test successful")
        return True
    except Exception as e:
        import traceback

        print(f"  ✗ Helpers test failed: {e}")
        print(traceback.format_exc())
        return False


def main():
    """すべてのテストを実行"""
    print("=" * 60)
    print("🧪 Experiment Tracking System - Basic Tests")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Database", test_database()))
    results.append(("Tracker", test_tracker()))
    results.append(("Helpers", test_helpers()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")

    print("=" * 60)

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
