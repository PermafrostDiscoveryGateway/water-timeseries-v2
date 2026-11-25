"""Guard: no new hard-coded ``id_geohash`` *selections* in ``src/``.

Issue #312 was the second pass at this bug: `LakeDataset.id_field` existed, but free
functions still indexed frames with the literal ``"id_geohash"``. This test fails when a
new selection appears, so the next one cannot be merged unnoticed.

What counts as a selection
--------------------------
* ``frame["id_geohash"]`` / ``frame["id_geohash"] == x``
* ``obj.id_geohash`` attribute access
* ``sel(...)``, ``dropna(...)``, ``set_index(...)``, ``groupby(...)``, ``transpose(...)``,
  ``drop(columns=...)``, ``read_parquet(filters=...)`` receiving ``"id_geohash"``

What does not
-------------
Defaults (``id_field: str = "id_geohash"``), output/contract column names, dict keys,
labels and docstrings. Those are legitimate and stay.

An allowlisted file must state *why* — the reason is the review hook.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "water_timeseries"

SELECTING_CALLS = {
    "sel",
    "isel",
    "dropna",
    "set_index",
    "groupby",
    "transpose",
    "drop",
    "read_parquet",
    "to_dataframe",
}

# path -> why this file may still name "id_geohash" when selecting
ALLOWLIST = {
    "scripts/precompute_nrt_monthly.py": "reads the archive zarr, whose id dim is the contract name",
    "scripts/plot_pipeline.py": "reads contract break parquets",
    "scripts/cli.py": "passes the contract column name to the contract-only dashboard",
    "utils/nrt_postprocessing.py": "joins contract parquets produced by this pipeline",
    "dashboard/map_viewer.py": "dashboard reads contract parquets/pmtiles and is geohash-only by design",
}


def _parents(tree):
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _is_selection(node, parent):
    target = parent.get(node)
    if isinstance(target, ast.Subscript):
        return True
    if isinstance(target, ast.Attribute):
        return True
    if isinstance(target, ast.keyword) and target.arg in {"dim", "columns", "filters", "by", "id_where"}:
        return True
    if isinstance(target, ast.Call):
        func = target.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        return name in SELECTING_CALLS
    if isinstance(target, (ast.Tuple, ast.List)):
        outer = parent.get(target)
        if isinstance(outer, ast.keyword) and outer.arg == "filters":
            return True
        if isinstance(outer, ast.List):
            grand = parent.get(outer)
            if isinstance(grand, ast.keyword) and grand.arg == "filters":
                return True
    return False


def _find_selections():
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parent = _parents(tree)
        lines = path.read_text(encoding="utf-8").splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or node.value != "id_geohash":
                continue
            if _is_selection(node, parent):
                hits.append(f"{rel}:{node.lineno}: {lines[node.lineno - 1].strip()}")
    return hits


def test_no_unlisted_hardcoded_id_selection():
    hits = _find_selections()
    unlisted = [h for h in hits if h.split(":", 1)[0] not in ALLOWLIST]
    assert not unlisted, (
        "Hard-coded 'id_geohash' selection(s) found. Use dataset.id_field, an id_field/id_column "
        "parameter, or select_feature_by_id():\n  " + "\n  ".join(unlisted)
    )


def test_allowlist_entries_still_needed():
    """A stale allowlist entry hides a real fix — prune it when the file is clean."""
    hits = _find_selections()
    present = {h.split(":", 1)[0] for h in hits}
    stale = sorted(set(ALLOWLIST) - present)
    assert not stale, f"Allowlist entries no longer needed (no selections found): {stale}"
