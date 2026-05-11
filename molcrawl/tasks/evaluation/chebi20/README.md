# ChEBI-20 evaluation

Phase 5 bidirectional evaluator:

- **molecule -> caption**: BLEU-4, ROUGE-L, exact match.
- **caption -> SMILES**: exact match, mean Levenshtein, SMILES validity.

Generation goes through `ModelAdapter.generate`.  Heavy NLP deps
(`nltk`, `rouge-score`, `rdkit`) are imported lazily so the evaluator
returns NaN when they are missing instead of failing outright.
