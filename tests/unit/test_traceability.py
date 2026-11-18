"""Spec <-> test traceability (conventions in specs/00-overview.md).

1. Every requirement ID defined in specs/*.md has at least one @pytest.mark.req("ID") test.
2. Every ID cited by a test is well formed, uses a known prefix and is defined in a spec.
3. The proposal coverage matrix in specs/00-overview.md maps every proposal phrase to defined IDs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"
OVERVIEW = SPECS / "00-overview.md"
ID = r"[A-Z]+-\d{2}"

PROPOSAL = (
    "For my final project I propose Outrun, an AI system that uses deep learning and real weather, radar and "
    "satellite data to forecast severe storms, tornadoes, flash floods, wildfire spread, icy roads and smoke, then "
    "uses A* search and optimization on real road maps to plan safe evacuation routes, shelter assignments, "
    "whole-city evacuations and emergency vehicle routes, demonstrated through an interactive map app."
)
EXPECTED_SPECS = {
    "01-data.md": {"DATA"},
    "02-grid-risk.md": {"GRID", "RISK", "HORIZON", "CAL"},
    "03-roads.md": {"ROAD"},
    "04-astar.md": {"ASTAR", "PLAN"},
    "05-shelters.md": {"SHEL"},
    "06-evacuation.md": {"EVAC"},
    "07-emergency.md": {"EMS"},
    "08-storm.md": {"STORM"},
    "09-tornado.md": {"TORN"},
    "10-flood.md": {"FLOOD"},
    "11-wildfire.md": {"FIRE"},
    "12-ice.md": {"ICE"},
    "13-smoke.md": {"SMOKE"},
    "14-app.md": {"APP", "CLI"},
    "15-evaluation.md": {"EVAL", "SPLIT"},
    "16-acceptance.md": {"AC"},
}


# ---------------------------------------------------------------- parsing helpers
def _section(text: str, heading: str) -> str:
    """Text of the level-2 section whose heading starts with ``heading`` (up to the next level-2 heading)."""
    m = re.search(rf"^##\s+{re.escape(heading)}.*?$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    assert m, f"section '## {heading}' not found in {OVERVIEW.name}"
    return m.group(1)


def _table_rows(section: str) -> list[list[str]]:
    """Cells of each Markdown table body row (header and separator rows dropped)."""
    rows = [ln.strip() for ln in section.splitlines() if ln.strip().startswith("|")]
    body = [r for r in rows[1:] if not re.fullmatch(r"\|[\s:|-]+\|", r)]
    return [[c.strip() for c in r.strip("|").split("|")] for r in body]


def _prefix_table() -> dict[str, set[str]]:
    """{spec file name: set of prefixes} from the 'Spec files and ID prefixes' table."""
    table = {}
    for cells in _table_rows(_section(OVERVIEW.read_text(encoding="utf-8"), "Spec files and ID prefixes")):
        table[cells[0].strip("`")] = {p.strip() for p in cells[1].split(",") if p.strip()}
    return table


def _known_prefixes() -> set[str]:
    return set().union(*_prefix_table().values())


def _defined_ids_by_spec() -> dict[str, list[str]]:
    """{spec file name: IDs it defines}, from '### ID' headings and '| **ID** |' acceptance rows."""
    out: dict[str, list[str]] = {}
    for f in sorted(SPECS.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        ids = re.findall(rf"^###\s+({ID})\b", text, flags=re.M)
        ids += re.findall(rf"^\|\s*\*\*({ID})\*\*", text, flags=re.M)
        out[f.name] = ids
    return out


def _spec_ids() -> set[str]:
    return {i for ids in _defined_ids_by_spec().values() for i in ids}


def _cited_ids() -> dict[str, set[str]]:
    """{raw string cited in req(...): test files citing it}, across all of tests/."""
    cited: dict[str, set[str]] = {}
    for f in (ROOT / "tests").rglob("*.py"):
        if f.resolve() == Path(__file__).resolve():
            continue
        for raw in re.findall(r"""\breq\(\s*["']([^"']*)["']""", f.read_text(encoding="utf-8")):
            cited.setdefault(raw, set()).add(str(f.relative_to(ROOT)))
    return cited


def _expand(token: str) -> list[str]:
    """'STORM-01..STORM-03' -> ['STORM-01', 'STORM-02', 'STORM-03']; 'ROAD-06' -> ['ROAD-06']."""
    token = token.strip().strip("`")
    m = re.fullmatch(r"([A-Z]+)-(\d{2})\.\.([A-Z]+)-(\d{2})", token)
    if m:
        p1, a, p2, b = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        assert p1 == p2 and a <= b, f"bad range in proposal matrix: {token}"
        return [f"{p1}-{i:02d}" for i in range(a, b + 1)]
    assert re.fullmatch(ID, token), f"malformed ID in proposal matrix: {token!r}"
    return [token]


def _matrix() -> list[tuple[str, list[str]]]:
    rows = []
    for cells in _table_rows(_section(OVERVIEW.read_text(encoding="utf-8"), "Proposal coverage matrix")):
        phrase, ids_cell = cells[0], cells[-1]
        ids = [i for tok in ids_cell.split(",") if tok.strip() for i in _expand(tok)]
        rows.append((phrase, ids))
    return rows


# ---------------------------------------------------------------- tests
def test_prefix_table_matches_fixed_assignment():
    assert _prefix_table() == EXPECTED_SPECS


def test_all_spec_files_exist():
    missing = sorted(name for name in EXPECTED_SPECS if not (SPECS / name).exists())
    assert not missing, f"Spec files missing: {missing}"


def test_specs_define_ids_only_with_their_own_prefixes():
    table = _prefix_table()
    wrong = []
    for name, ids in _defined_ids_by_spec().items():
        allowed = table.get(name, set())  # 00-overview defines nothing
        wrong += [f"{name}: {i}" for i in ids if i.split("-")[0] not in allowed]
    assert not wrong, f"IDs defined in the wrong spec file (see prefix table): {wrong}"


def test_no_duplicate_requirement_ids():
    counts = Counter(i for ids in _defined_ids_by_spec().values() for i in ids)
    dupes = sorted(i for i, n in counts.items() if n > 1)
    assert not dupes, f"Requirement IDs defined more than once: {dupes}"


def test_every_spec_requirement_has_a_test():
    missing = sorted(_spec_ids() - set(_cited_ids()))
    assert not missing, f"Requirements without tests: {missing}"


def test_every_test_id_exists_in_specs():
    cited = _cited_ids()
    malformed = sorted(
        f"{raw!r} in {sorted(files)}" for raw, files in cited.items() if not re.fullmatch(ID, raw)
    )
    assert not malformed, f"Malformed requirement IDs in req(...): {malformed}"
    known = _known_prefixes()
    bad_prefix = sorted(f"{raw} in {sorted(f)}" for raw, f in cited.items() if raw.split("-")[0] not in known)
    assert not bad_prefix, f"Tests cite IDs with unknown prefixes: {bad_prefix}"
    unknown = sorted(f"{raw} in {sorted(f)}" for raw, f in cited.items() if raw not in _spec_ids())
    assert not unknown, f"Tests reference undefined requirement IDs: {unknown}"


def test_proposal_quote_is_exact():
    assert PROPOSAL in OVERVIEW.read_text(encoding="utf-8")


def test_proposal_matrix():
    rows = _matrix()
    assert len(rows) >= 19, f"proposal matrix has only {len(rows)} rows"
    phrases = [p for p, _ in rows]
    # every listed phrase (except the professor's "any US city" follow-up) is made of words from the proposal
    # ("real radar data" abbreviates "real weather, radar and satellite data")
    words = set(re.findall(r"[\w*-]+", PROPOSAL))
    not_in_proposal = [
        p for p in phrases if not p.startswith("any US city") and not set(re.findall(r"[\w*-]+", p)) <= words
    ]
    assert not not_in_proposal, f"Matrix phrases not found in the proposal text: {not_in_proposal}"
    empty = [p for p, ids in rows if not ids]
    assert not empty, f"Proposal phrases without requirement IDs: {empty}"
    defined, tested = _spec_ids(), set(_cited_ids())
    undefined = sorted({f"{i} ({p})" for p, ids in rows for i in ids if i not in defined})
    assert not undefined, f"Matrix IDs not defined in any spec: {undefined}"
    untested = sorted({i for _, ids in rows for i in ids if i not in tested})
    assert not untested, f"Matrix IDs without a @pytest.mark.req test: {untested}"


def test_matrix_covers_every_proposal_item():
    """Each graded item of the proposal sentence has its own matrix row."""
    required = [
        "deep learning", "real weather data", "real radar data", "real satellite data",
        "forecast severe storms", "tornadoes", "flash floods", "wildfire spread", "icy roads", "smoke",
        "A* search", "optimization", "real road maps", "safe evacuation routes", "shelter assignments",
        "whole-city evacuations", "emergency vehicle routes", "interactive map app",
    ]  # fmt: skip
    phrases = {p for p, _ in _matrix()}
    missing = [r for r in required if r not in phrases]
    assert not missing, f"Proposal items without a matrix row: {missing}"


def test_unit_test_file_ownership():
    """Every tests/unit/test_*.py file (except this one) matches the 'Unit test file(s)' patterns of exactly one
    spec row, so area work stays file-disjoint as the overview promises."""
    import fnmatch

    rows = _table_rows(_section(OVERVIEW.read_text(encoding="utf-8"), "Spec files and ID prefixes"))
    patterns = {cells[0].strip("`"): re.findall(r"`([^`]+)`", cells[3]) for cells in rows}
    problems = []
    for f in sorted((ROOT / "tests" / "unit").glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue
        rel = f.relative_to(ROOT).as_posix()
        owners = [spec for spec, pats in patterns.items() if any(fnmatch.fnmatch(rel, p) for p in pats)]
        if len(owners) != 1:
            problems.append(f"{rel}: owned by {owners or 'no spec'}")
    assert not problems, "\n".join(problems)
