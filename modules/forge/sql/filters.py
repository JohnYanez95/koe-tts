"""Safe parameterized SQL filter parser.

This module provides SQL-injection-safe parsing of user-provided filter
expressions into parameterized SQL fragments + bind values.

Supported Filter Grammar
------------------------
Comparison:  col = 'value'  |  col >= 10  |  col != 'foo'
NULL check:  col is null    |  col is not null
IN list:     col in ('a', 'b', 'c')
LIKE:        col like '%pattern%'  |  col ilike 'case_insensitive'

Operators:   =, !=, <, >, <=, >=, LIKE, ILIKE, IN, IS NULL, IS NOT NULL

Value Syntax
------------
- Strings MUST be quoted: 'value' or "value" (barewords are rejected)
- Numbers: 42, -3.14, 0.5
- Booleans: true, false
- Null: null, none

DoS Limits
----------
- MAX_IN_LIST: {MAX_IN_LIST} items per IN clause
- MAX_FILTERS: {MAX_FILTERS} filters per query
- MAX_FILTER_LENGTH: {MAX_FILTER_LENGTH} chars per filter expression

Public API
----------
- parse_filter(expr, allowed_cols, schema_map) -> ParsedFilter
- build_where(filters, any_filters, ...) -> (sql, params)
- parse_columns(columns, schema_map, allowed_cols) -> str
- validate_ident(name, kind, allowed) -> str
- quote_ident(name) -> str  (DuckDB double-quote)
- quote_spark_ident(name) -> str  (Spark backtick-quote)
- safe_sql_string(value) -> str  (escape for non-parameterisable literals)
- schema_map_from_columns(columns) -> dict
- FilterParseError: raised on invalid input

Engine Compatibility
-------------------
Generated SQL uses '?' placeholders (DuckDB, SQLite style).
For Postgres ($1, $2), a thin wrapper can renumber placeholders.

Example
-------
>>> pf = parse_filter("split = 'train'", allowed_cols={"split"})
>>> pf.sql, pf.params
('"split" = ?', ['train'])

>>> build_where(filters=["split='train'"], any_filters=["speaker='a'", "speaker='b'"])
('WHERE "split" = ? AND ("speaker" = ? OR "speaker" = ?)', ['train', 'a', 'b'])

Identifier Policy
-----------------
Dataset, table, and column names must be lowercase snake_case: [a-z][a-z0-9_]*
This is stricter than what DuckDB/Spark allow, but ensures portability and safety.

.. warning::
   **DO NOT inline user-supplied strings into SQL.**

   - For filter values: use ``build_where()`` which returns ``(sql, params)``
     and pass ``params`` to ``conn.execute(sql, params)``.
   - For identifiers: use ``validate_ident()`` + ``quote_ident()`` or
     ``quote_spark_ident()``.
   - For paths in non-parameterisable contexts (e.g., ``delta_scan('...')``):
     use ``safe_sql_string()``.

   If you're tempted to do ``f"... WHERE col = '{value}'"`` — stop and use
   this module instead.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ── Tunables ────────────────────────────────────────────────────────────
MAX_IN_LIST: int = 1_000
MAX_FILTERS: int = 50
MAX_FILTER_LENGTH: int = 10_000

SAFE_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


class FilterParseError(ValueError):
    pass


# ── Identifier helpers ──────────────────────────────────────────────────

def validate_ident(
    name: str,
    kind: str = "identifier",
    allowed: set[str] | None = None,
) -> str:
    if not SAFE_IDENT.fullmatch(name):
        raise FilterParseError(
            f"Invalid {kind}: {name!r} (expected [a-z][a-z0-9_]*)"
        )
    if allowed is not None and name not in allowed:
        raise FilterParseError(f"Unknown {kind}: {name!r}")
    return name


def canonicalize_column(
    name: str,
    schema_map: Mapping[str, str] | None = None,
    allowed: set[str] | None = None,
) -> str:
    """Resolve a user-supplied column name to its canonical form.

    If *schema_map* is provided (lowercase → actual), accept any case and
    return the canonical lowercase key.  Otherwise fall back to strict
    ``validate_ident`` behaviour.
    """
    if schema_map is not None:
        key = name.strip().lower()
        if key not in schema_map:
            raise FilterParseError(
                f"Unknown column: {name!r}. "
                f"Available: {sorted(schema_map)}"
            )
        return key  # canonical lowercase form

    # No schema map → strict regex + optional allowlist
    return validate_ident(name, "column", allowed)


def quote_ident(name: str) -> str:
    """Double-quote an identifier for SQL.

    Defense-in-depth: even though ``validate_ident`` forbids special
    characters today, we independently reject embedded quotes so that a
    future relaxation of the identifier regex cannot produce injection.
    """
    if '"' in name:
        raise FilterParseError(
            f"Identifier contains illegal quote character: {name!r}"
        )
    return f'"{name}"'


# ── Value parsing ───────────────────────────────────────────────────────

def split_top_level_commas(s: str) -> list[str]:
    """Split on commas, but ignore commas inside quotes or parentheses."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
        else:
            buf.append(ch)

    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out


def parse_literal(token: str):
    """Parse a value token into a Python literal to bind as a parameter.

    Only numbers, booleans, null, and *quoted* strings are accepted.
    Unquoted barewords are rejected to prevent accidental expression
    injection if parameterisation is ever bypassed downstream.
    """
    t = token.strip()
    if not t:
        raise FilterParseError("Empty value")

    low = t.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"

    # Integer
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    # Float
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", t):
        return float(t)

    # Quoted strings → safe parse via ast.literal_eval
    if (t.startswith("'") and t.endswith("'")) or (
        t.startswith('"') and t.endswith('"')
    ):
        try:
            v = ast.literal_eval(t)
        except Exception as e:
            raise FilterParseError(f"Bad quoted literal: {t!r}") from e
        if not isinstance(v, (str, int, float, bool, type(None))):
            raise FilterParseError(
                f"Unsupported literal type: {type(v).__name__}"
            )
        return v

    # ── No bareword fallback ────────────────────────────────────────
    raise FilterParseError(
        f"Unquoted string literal {t!r}. "
        f"Quote strings like 'value' or \"value\"."
    )


# ── Filter grammar ──────────────────────────────────────────────────────

_NULL_RE = re.compile(
    r"^(?P<col>[a-z][a-z0-9_]*)\s+is\s+(?P<not>not\s+)?null$", re.I
)
_IN_RE = re.compile(
    r"^(?P<col>[a-z][a-z0-9_]*)\s+in\s*\((?P<list>.*)\)$", re.I
)
_CMP_RE = re.compile(
    r"^(?P<col>[a-z][a-z0-9_]*)\s*(?P<op>>=|<=|!=|=|>|<)\s*(?P<val>.+)$",
    re.I,
)
_LIKE_RE = re.compile(
    r"^(?P<col>[a-z][a-z0-9_]*)\s+(?P<op>like|ilike)\s+(?P<val>.+)$", re.I
)


@dataclass(frozen=True)
class ParsedFilter:
    sql: str
    params: list


def parse_filter(
    expr: str,
    allowed_cols: set[str] | None = None,
    schema_map: Mapping[str, str] | None = None,
) -> ParsedFilter:
    if len(expr) > MAX_FILTER_LENGTH:
        raise FilterParseError(
            f"Filter exceeds max length ({len(expr)} > {MAX_FILTER_LENGTH})"
        )

    e = expr.strip()
    if not e:
        raise FilterParseError("Empty filter")

    def _col(raw: str) -> str:
        return quote_ident(
            canonicalize_column(raw, schema_map=schema_map, allowed=allowed_cols)
        )

    # ── NULL check ──────────────────────────────────────────────────
    m = _NULL_RE.match(e)
    if m:
        col = _col(m.group("col"))
        not_ = bool(m.group("not"))
        return ParsedFilter(f"{col} IS {'NOT ' if not_ else ''}NULL", [])

    # ── IN list ─────────────────────────────────────────────────────
    m = _IN_RE.match(e)
    if m:
        col = _col(m.group("col"))
        items = split_top_level_commas(m.group("list"))
        if not items:
            raise FilterParseError("IN() must contain at least one value")
        if len(items) > MAX_IN_LIST:
            raise FilterParseError(
                f"IN list too large ({len(items)}). Max {MAX_IN_LIST}."
            )
        vals = [parse_literal(x) for x in items]
        placeholders = ", ".join(["?"] * len(vals))
        return ParsedFilter(f"{col} IN ({placeholders})", vals)

    # ── LIKE / ILIKE ────────────────────────────────────────────────
    m = _LIKE_RE.match(e)
    if m:
        col = _col(m.group("col"))
        op = m.group("op").upper()
        val = parse_literal(m.group("val"))
        if not isinstance(val, str):
            raise FilterParseError(f"{op} requires a string pattern")
        return ParsedFilter(f"{col} {op} ?", [val])

    # ── Comparison ──────────────────────────────────────────────────
    m = _CMP_RE.match(e)
    if m:
        col = _col(m.group("col"))
        op = m.group("op")
        val = parse_literal(m.group("val"))
        return ParsedFilter(f"{col} {op} ?", [val])

    raise FilterParseError(f"Unsupported filter syntax: {expr!r}")


# ── WHERE clause builders ───────────────────────────────────────────────

def _parse_filter_list(
    filters: Sequence[str],
    allowed_cols: set[str] | None,
    schema_map: Mapping[str, str] | None,
) -> tuple[list[str], list]:
    """Parse a sequence of filter expressions into SQL fragments + params."""
    if len(filters) > MAX_FILTERS:
        raise FilterParseError(
            f"Too many filters ({len(filters)}). Max {MAX_FILTERS}."
        )
    fragments: list[str] = []
    params: list = []
    for f in filters:
        pf = parse_filter(f, allowed_cols=allowed_cols, schema_map=schema_map)
        fragments.append(pf.sql)
        params.extend(pf.params)
    return fragments, params


def build_where(
    filters: Sequence[str] | None = None,
    any_filters: Sequence[str] | None = None,
    allowed_cols: set[str] | None = None,
    schema_map: Mapping[str, str] | None = None,
) -> tuple[str, list]:
    """Build a WHERE clause from AND filters and optional OR (any) filters.

    Parameters
    ----------
    filters : list[str] | None
        Predicates combined with AND.
    any_filters : list[str] | None
        Predicates combined with OR, then ANDed with *filters*.
    allowed_cols : set[str] | None
        Strict allowlist of column names (lowercase).
    schema_map : dict[str, str] | None
        Optional ``{lowercase: canonical}`` column mapping.  When provided,
        columns are resolved case-insensitively against the map and
        *allowed_cols* is ignored.

    Returns
    -------
    (where_sql, params) where *where_sql* includes a leading ``WHERE ``
    or is the empty string if no predicates were supplied.

    Examples
    --------
    >>> build_where(
    ...     filters=["split='train'"],
    ...     any_filters=["speaker='jvs001'", "speaker='jvs002'"],
    ... )
    ("WHERE \"split\" = ? AND (\"speaker\" = ? OR \"speaker\" = ?)",
     ['train', 'jvs001', 'jvs002'])
    """
    clauses: list[str] = []
    all_params: list = []

    # ── AND filters ─────────────────────────────────────────────────
    if filters:
        frags, params = _parse_filter_list(filters, allowed_cols, schema_map)
        clauses.extend(frags)
        all_params.extend(params)

    # ── OR (any) filters ────────────────────────────────────────────
    if any_filters:
        frags, params = _parse_filter_list(any_filters, allowed_cols, schema_map)
        if frags:
            or_clause = " OR ".join(frags)
            # Parenthesise so it ANDs cleanly with the rest
            clauses.append(f"({or_clause})")
            all_params.extend(params)

    if not clauses:
        return "", []

    return "WHERE " + " AND ".join(clauses), all_params


# ── Schema-map helper (for Option B canonicalization) ───────────────────

# ── Column list parsing ─────────────────────────────────────────────────

MAX_COLUMNS: int = 200


def parse_columns(
    columns: str,
    schema_map: Mapping[str, str] | None = None,
    allowed_cols: set[str] | None = None,
) -> str:
    """Validate and quote a comma-separated column list.

    Accepts ``"*"`` as a pass-through.  Otherwise each column is validated
    against *schema_map* (preferred) or *allowed_cols* and double-quoted.

    >>> parse_columns("split, duration_sec", allowed_cols={"split", "duration_sec"})
    '"split", "duration_sec"'
    """
    c = columns.strip()
    if c == "*":
        return "*"

    parts = [x.strip() for x in split_top_level_commas(c)]
    if not parts:
        raise FilterParseError("Empty columns")
    if len(parts) > MAX_COLUMNS:
        raise FilterParseError(
            f"Too many columns ({len(parts)}). Max {MAX_COLUMNS}."
        )

    safe: list[str] = []
    for col in parts:
        canonical = canonicalize_column(
            col, schema_map=schema_map, allowed=allowed_cols
        )
        safe.append(quote_ident(canonical))
    return ", ".join(safe)


# ── Safe string-literal helper ──────────────────────────────────────────


def safe_sql_string(value: str) -> str:
    """Escape a value for use as a SQL single-quoted string literal.

    This is a last-resort helper for contexts where bind parameters
    aren't supported (e.g. ``delta_scan('...')``, Spark DDL
    ``LOCATION '...'``).  It escapes embedded single quotes by doubling
    them (standard SQL escaping) and rejects null bytes.

    >>> safe_sql_string("/lake/silver/jsut/utterances")
    "'/lake/silver/jsut/utterances'"
    >>> safe_sql_string("it's a path")
    "'it''s a path'"
    """
    if "\x00" in value:
        raise FilterParseError("String literal contains null byte")
    return "'" + value.replace("'", "''") + "'"


# ── Identifier helpers for Spark SQL (backtick-quoted) ──────────────────

SAFE_SPARK_IDENT = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def quote_spark_ident(name: str) -> str:
    """Backtick-quote an identifier for Spark SQL.

    Validates that the name matches a safe pattern and contains no
    backticks, then wraps it.

    >>> quote_spark_ident("gold_koe")
    '`gold_koe`'
    """
    if not SAFE_SPARK_IDENT.fullmatch(name):
        raise FilterParseError(
            f"Invalid Spark identifier: {name!r} "
            f"(expected [a-zA-Z][a-zA-Z0-9_]*)"
        )
    if "`" in name:
        raise FilterParseError(
            f"Spark identifier contains illegal backtick: {name!r}"
        )
    return f"`{name}`"


# ── Schema-map helper (for Option B canonicalization) ───────────────────


def schema_map_from_columns(columns: Sequence[str]) -> dict[str, str]:
    """Build a ``{lowercase: original}`` mapping from real table columns.

    Use with DuckDB's ``DESCRIBE`` or ``information_schema.columns`` to
    enable case-insensitive, allowlisted column references in filters.

    >>> schema_map_from_columns(["SpeakerID", "split", "Duration_s"])
    {'speakerid': 'SpeakerID', 'split': 'split', 'duration_s': 'Duration_s'}
    """
    return {c.lower(): c for c in columns}
