# ChemLLMBench evaluation

Phase 5 evaluator covering the nine ChemLLMBench sub-tasks.  Each
sub-task is a prompt / answer JSONL; the evaluator generates with the
adapter and scores the output through a task-type-specific metric:

| sub-task | metric |
| --- | --- |
| name_conversion / property_prediction / smiles_understanding | exact match |
| reaction_prediction / retrosynthesis / text_guided_generation / molecule_design | SMILES pack (exact, Levenshtein, validity) |
| yield_prediction | RMSE / MAE |
| molecule_captioning | text pack (BLEU, ROUGE, exact) |
