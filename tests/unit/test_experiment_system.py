"""
Basic test of experiment management system
"""

import sys
import time
from pathlib import Path

# add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_imports():
    """Module import test"""
    print("Testing imports...")
    try:
        from molcrawl.core.tracking import (  # noqa: F401
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
    """Database connection test"""
    print("\nTesting database...")
    try:
        import os
        import tempfile

        from molcrawl.core.tracking.database import ExperimentDatabase

        # use temporary files
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        db = ExperimentDatabase(db_path)
        stats = db.get_statistics()
        assert stats["total_experiments"] == 0

        # cleanup
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
    """Tracker basic functionality test"""
    print("\nTesting tracker...")
    try:
        import os

        # Temporary database for testing
        import tempfile

        from molcrawl.core.tracking import (
            DatasetType,
            ExperimentTracker,
            ExperimentType,
            ModelType,
        )

        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        tracker = ExperimentTracker(db_path)

        # Create experiment
        exp_id = tracker.start_experiment(
            name="Test Experiment",
            experiment_type=ExperimentType.TRAINING,
            model_type=ModelType.GPT2,
            dataset_type=DatasetType.PROTEIN_SEQUENCE,
        )
        print(f"    Created experiment: {exp_id}")

        # add step
        tracker.start_step(exp_id, "test_step", "Test Step")
        time.sleep(0.1)
        tracker.complete_step(exp_id, "test_step")
        print("    Added and completed step")

        # add log
        tracker.log(exp_id, "INFO", "Test log message")
        print("    Added log entry")

        # Experiment completed
        tracker.complete_experiment(exp_id, metrics={"accuracy": 0.95})
        print("    Completed experiment")

        # Acquisition test
        experiment = tracker.get_experiment(exp_id)
        assert experiment is not None
        assert len(experiment.steps) == 1
        assert len(experiment.logs) >= 2  # start + test log
        assert experiment.metrics["accuracy"] == 0.95
        print("    Retrieved and verified experiment")

        # cleanup
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
    """Helper function test"""
    print("\nTesting helpers...")
    try:
        import tempfile

        from molcrawl.core.tracking import (
            DatasetType,
            ExperimentType,
            ModelType,
        )
        from molcrawl.core.tracking.helpers import experiment_context

        temp_dir = tempfile.mkdtemp()

        # Context manager test
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

        # cleanup
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
    """Run all tests"""
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
