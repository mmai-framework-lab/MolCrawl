"""A caller that tokenises for itself does not have to produce a tokenizer.

AutoTokenizer.from_pretrained wants a directory holding a tokenizer config. compounds
has none: the vocabulary is a bare vocab.txt read by CompoundsTokenizer, and the
checkpoints carry no tokenizer files. The compounds probe therefore tokenises itself --
it has to, since build_encoder's encode splits with list(s) and 579 of the 612 compounds
tokens are multi-character -- and passing the vocab.txt path only produced

    OSError: Incorrect path_or_model_id: 'assets/molecules/vocab.txt'

which is what took out the first launch of the downstream probe (jobs 206657-206687).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "molcrawl" / "models" / "_representations.py"
PROBE = ROOT / "scripts" / "compounds_downstream_eval.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def test_the_tokenizer_is_loaded_only_when_a_path_was_given():
    """An empty path must not reach AutoTokenizer; a real one must still work."""
    tree = ast.parse(_source(MODULE))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_encoder")
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
    assert guards, "build_encoder has no conditional around the tokenizer load"
    # The load sits inside a branch, not at the top level of the body.
    top_level_calls = [
        n for stmt in fn.body if not isinstance(stmt, ast.If)
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "from_pretrained"
    ]
    assert not top_level_calls, "from_pretrained still runs unconditionally"


def test_encode_refuses_rather_than_guessing():
    """Both families must say why encode is unavailable, not fail on tok being None."""
    src = _source(MODULE)
    assert src.count("build_encoder was called without a tokenizer_path") == 2, (
        "each family's encode needs its own refusal")
    assert src.count("if tok is None:") == 2


def test_the_compounds_probe_asks_for_no_tokenizer():
    src = _source(PROBE)
    assert "build_encoder(\n        model_path, None, arch, device," in src, (
        "the probe must pass no tokenizer; it tokenises with CompoundsTokenizer")
    # And it must still be the one doing the tokenising.
    assert "CompoundsTokenizer" in src


def test_the_genome_probe_still_requires_one():
    """This is opt-in: a caller that uses encode must still supply a tokenizer."""
    src = _source(ROOT / "scripts" / "clinvar_linear_probe.py")
    assert "--tokenizer is required when scoring model representations" in src
