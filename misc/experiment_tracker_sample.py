from src.experiment_tracker import DatasetType, ExperimentType, ModelType
from src.experiment_tracker.helpers import experiment_context

with experiment_context(
    name="あなたの実験名",
    experiment_type=ExperimentType.TRAINING,
    model_type=ModelType.GPT2,
    dataset_type=DatasetType.PROTEIN_SEQUENCE,
) as exp:
    # あなたのコード
    exp.start_step("step1", "ステップ1")
    # 処理...
    exp.complete_step("step1")
    exp.add_metric("accuracy", 0.95)


print("Experiment completed with metrics:", exp.metrics)
