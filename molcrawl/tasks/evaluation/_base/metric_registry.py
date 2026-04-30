"""Registry of reusable evaluation metrics.

The evaluators in ``molcrawl.tasks.evaluation.*`` select metrics by name
through :data:`default_registry`.  The registry covers the four metric
families called out in the evaluator implementation plan:

* perplexity (decoder likelihood scoring)
* classification (accuracy, F1, AUROC, AUPRC, MCC)
* regression (RMSE, MAE, R^2, Spearman, Pearson)
* generation quality (validity, uniqueness, novelty, internal diversity)

Heavy optional dependencies (``scikit-learn``, ``scipy``, ``rdkit``) are
imported lazily inside each metric function so that registering them is
cheap and the module can be imported in environments that only need a
subset of metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

MetricFn = Callable[..., float]


@dataclass
class MetricSpec:
    """Metadata describing a registered metric."""

    name: str
    family: str  # "perplexity" | "classification" | "regression" | "generation"
    higher_is_better: bool
    fn: MetricFn
    description: str = ""


class MetricRegistry:
    """In-memory registry keyed by metric name."""

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricSpec] = {}

    def register(
        self,
        name: str,
        family: str,
        fn: MetricFn,
        higher_is_better: bool,
        description: str = "",
    ) -> None:
        if name in self._metrics:
            raise ValueError(f"Metric already registered: {name!r}")
        self._metrics[name] = MetricSpec(
            name=name,
            family=family,
            higher_is_better=higher_is_better,
            fn=fn,
            description=description,
        )

    def get(self, name: str) -> MetricSpec:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown metric {name!r}. Registered: {sorted(self._metrics)}"
            ) from exc

    def compute(self, name: str, *args: Any, **kwargs: Any) -> float:
        spec = self.get(name)
        return float(spec.fn(*args, **kwargs))

    def list(self, family: Optional[str] = None) -> List[str]:
        if family is None:
            return sorted(self._metrics)
        return sorted(n for n, s in self._metrics.items() if s.family == family)

    def __contains__(self, name: str) -> bool:
        return name in self._metrics


# ---------------------------------------------------------------------------
# Reference metric implementations
# ---------------------------------------------------------------------------


def _as_float_array(values: Sequence[float]):
    import numpy as np

    return np.asarray(values, dtype=float)


def _as_int_array(values: Sequence[int]):
    import numpy as np

    return np.asarray(values, dtype=int)


def _perplexity(log_likelihood_per_token: float) -> float:
    """Convert mean per-token NLL (as a positive number) into perplexity.

    Callers are expected to pass the mean negative log-likelihood per
    token.  Signs follow the convention ``ppl = exp(mean_nll)``.
    """
    return float(math.exp(log_likelihood_per_token))


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(_as_int_array(y_true), _as_int_array(y_pred)))


def _f1_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(_as_int_array(y_true), _as_int_array(y_pred), average="binary"))


def _f1_macro(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(_as_int_array(y_true), _as_int_array(y_pred), average="macro"))


def _auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(_as_int_array(y_true), _as_float_array(y_score)))


def _auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(_as_int_array(y_true), _as_float_array(y_score)))


def _mcc(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(_as_int_array(y_true), _as_int_array(y_pred)))


def _rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    import numpy as np

    y_true_arr = _as_float_array(y_true)
    y_pred_arr = _as_float_array(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def _mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    import numpy as np

    return float(np.mean(np.abs(_as_float_array(y_true) - _as_float_array(y_pred))))


def _r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    from sklearn.metrics import r2_score

    return float(r2_score(_as_float_array(y_true), _as_float_array(y_pred)))


def _spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(_as_float_array(y_true), _as_float_array(y_pred))
    return float(rho)


def _pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    from scipy.stats import pearsonr

    r, _ = pearsonr(_as_float_array(y_true), _as_float_array(y_pred))
    return float(r)


# ----- Generation metrics -----


def _canonical_smiles(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _validity(generated: Sequence[str]) -> float:
    if not generated:
        return 0.0
    valid = sum(1 for s in generated if _canonical_smiles(s) is not None)
    return valid / len(generated)


def _uniqueness(generated: Sequence[str]) -> float:
    valid = [c for c in (_canonical_smiles(s) for s in generated) if c is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def _novelty(generated: Sequence[str], reference: Sequence[str]) -> float:
    valid = [c for c in (_canonical_smiles(s) for s in generated) if c is not None]
    if not valid:
        return 0.0
    ref_set = {c for c in (_canonical_smiles(s) for s in reference) if c is not None}
    novel = sum(1 for s in valid if s not in ref_set)
    return novel / len(valid)


def _internal_diversity(generated: Sequence[str]) -> float:
    """Mean pairwise Tanimoto distance over Morgan fingerprints.

    Returns ``0.0`` when RDKit is unavailable or when fewer than two valid
    molecules remain, so that the metric degrades gracefully in CI.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return 0.0

    mols = []
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    n = len(fps)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total += 1.0 - sim
            pairs += 1
    return total / pairs if pairs else 0.0


def build_default_registry() -> MetricRegistry:
    registry = MetricRegistry()

    # perplexity
    registry.register(
        "perplexity", "perplexity", _perplexity, higher_is_better=False,
        description="exp(mean per-token NLL)",
    )

    # classification
    registry.register("accuracy", "classification", _accuracy, True)
    registry.register("f1_binary", "classification", _f1_binary, True)
    registry.register("f1_macro", "classification", _f1_macro, True)
    registry.register("auroc", "classification", _auroc, True)
    registry.register("auprc", "classification", _auprc, True)
    registry.register("mcc", "classification", _mcc, True)

    # regression
    registry.register("rmse", "regression", _rmse, higher_is_better=False)
    registry.register("mae", "regression", _mae, higher_is_better=False)
    registry.register("r2", "regression", _r2, True)
    registry.register("spearman", "regression", _spearman, True)
    registry.register("pearson", "regression", _pearson, True)

    # generation quality
    registry.register("validity", "generation", _validity, True)
    registry.register("uniqueness", "generation", _uniqueness, True)
    registry.register("novelty", "generation", _novelty, True)
    registry.register("internal_diversity", "generation", _internal_diversity, True)

    return registry


default_registry: MetricRegistry = build_default_registry()
