# test_filters.py
"""Test suite for hardened filters.py

Covers:
  1. Bareword rejection (no more silent string coercion)
  2. quote_ident defense-in-depth
  3. DoS caps (IN list size, filter count, filter length)
  4. Operator edge cases (==, = =, etc.)
  5. Injection attempts that should be inert
  6. any_filters (OR) support
  7. Schema-map canonicalization
  8. LIKE wildcard patterns (logical, not injection)
"""

import pytest

from modules.forge.sql.filters import (
    MAX_COLUMNS,
    MAX_FILTER_LENGTH,
    MAX_FILTERS,
    MAX_IN_LIST,
    FilterParseError,
    build_where,
    canonicalize_column,
    parse_columns,
    parse_filter,
    parse_literal,
    quote_ident,
    quote_spark_ident,
    safe_sql_string,
    schema_map_from_columns,
    validate_ident,
)

# ════════════════════════════════════════════════════════════════════════
# 1. Bareword rejection
# ════════════════════════════════════════════════════════════════════════

class TestBarewordRejection:
    """Unquoted non-numeric, non-boolean, non-null tokens must be rejected."""

    def test_rejects_unquoted_string(self):
        with pytest.raises(FilterParseError, match="Unquoted string literal"):
            parse_literal("train")

    def test_rejects_sql_keyword_bareword(self):
        with pytest.raises(FilterParseError, match="Unquoted string literal"):
            parse_literal("CURRENT_DATE")

    def test_rejects_column_name_lookalike(self):
        with pytest.raises(FilterParseError, match="Unquoted string literal"):
            parse_literal("other_column")

    def test_allows_quoted_string(self):
        assert parse_literal("'train'") == "train"
        assert parse_literal('"train"') == "train"

    def test_allows_numbers(self):
        assert parse_literal("42") == 42
        assert parse_literal("-3.14") == -3.14

    def test_allows_booleans(self):
        assert parse_literal("true") is True
        assert parse_literal("false") is False

    def test_allows_null(self):
        assert parse_literal("null") is None
        assert parse_literal("None") is None

    def test_filter_rejects_bareword_value(self):
        with pytest.raises(FilterParseError, match="Unquoted string literal"):
            parse_filter("col = train", allowed_cols={"col"})

    def test_filter_accepts_quoted_value(self):
        pf = parse_filter("col = 'train'", allowed_cols={"col"})
        assert pf.params == ["train"]


# ════════════════════════════════════════════════════════════════════════
# 2. quote_ident defense-in-depth
# ════════════════════════════════════════════════════════════════════════

class TestQuoteIdent:
    def test_normal_ident(self):
        assert quote_ident("col") == '"col"'

    def test_rejects_embedded_quote(self):
        with pytest.raises(FilterParseError, match="illegal quote"):
            quote_ident('col"--')

    def test_validate_ident_already_blocks_quotes(self):
        """Belt-and-suspenders: validate_ident rejects before quote_ident."""
        with pytest.raises(FilterParseError):
            validate_ident('col"injection')


# ════════════════════════════════════════════════════════════════════════
# 3. DoS caps
# ════════════════════════════════════════════════════════════════════════

class TestDoSCaps:
    def test_in_list_over_max(self):
        huge = ", ".join([str(i) for i in range(MAX_IN_LIST + 1)])
        with pytest.raises(FilterParseError, match="IN list too large"):
            parse_filter(f"col in ({huge})", allowed_cols={"col"})

    def test_in_list_at_max(self):
        """Exactly at the cap should succeed."""
        items = ", ".join([str(i) for i in range(MAX_IN_LIST)])
        pf = parse_filter(f"col in ({items})", allowed_cols={"col"})
        assert len(pf.params) == MAX_IN_LIST

    def test_filter_count_over_max(self):
        filters = ["col = 1"] * (MAX_FILTERS + 1)
        with pytest.raises(FilterParseError, match="Too many filters"):
            build_where(filters=filters, allowed_cols={"col"})

    def test_filter_length_over_max(self):
        long_val = "'" + "a" * (MAX_FILTER_LENGTH + 1) + "'"
        with pytest.raises(FilterParseError, match="max length"):
            parse_filter(f"col = {long_val}", allowed_cols={"col"})


# ════════════════════════════════════════════════════════════════════════
# 4. Operator edge cases
# ════════════════════════════════════════════════════════════════════════

class TestOperatorEdgeCases:
    def test_double_equals_rejected(self):
        """col==5 matches _CMP_RE as col = '= 5', then parse_literal rejects the bareword."""
        with pytest.raises(FilterParseError):
            parse_filter("col == 5", allowed_cols={"col"})

    def test_spaced_equals_rejected(self):
        """col = = 5 also matches _CMP_RE as col = '= 5', same rejection path."""
        with pytest.raises(FilterParseError):
            parse_filter("col = = 5", allowed_cols={"col"})

    def test_not_equals(self):
        pf = parse_filter("col != 5", allowed_cols={"col"})
        assert pf.sql == '"col" != ?'
        assert pf.params == [5]

    def test_gte_negative_float(self):
        pf = parse_filter("col >= -1.2", allowed_cols={"col"})
        assert pf.sql == '"col" >= ?'
        assert pf.params == [-1.2]

    @pytest.mark.parametrize("op", [">=", "<=", "!=", "=", ">", "<"])
    def test_all_comparison_ops(self, op):
        pf = parse_filter(f"col {op} 10", allowed_cols={"col"})
        assert pf.sql == f'"col" {op} ?'
        assert pf.params == [10]


# ════════════════════════════════════════════════════════════════════════
# 5. Injection attempts (should be inert or rejected)
# ════════════════════════════════════════════════════════════════════════

class TestInjectionAttempts:
    def test_or_injection_in_quoted_value(self):
        """Quoted value containing OR 1=1 is just a literal string param."""
        pf = parse_filter("col = 'foo OR 1=1 --'", allowed_cols={"col"})
        assert pf.params == ["foo OR 1=1 --"]
        assert "OR" not in pf.sql

    def test_unquoted_injection_rejected(self):
        """Without quotes, the old bareword path would have let this through
        as a string param. Now it's rejected outright."""
        with pytest.raises(FilterParseError):
            parse_filter("col = foo OR 1=1 --", allowed_cols={"col"})

    def test_column_injection_rejected(self):
        with pytest.raises(FilterParseError):
            parse_filter('"col"; DROP TABLE--' + " = 1")

    def test_unknown_column_rejected(self):
        with pytest.raises(FilterParseError, match="Unknown column"):
            parse_filter("secret = 1", allowed_cols={"col"})

    def test_null_injection_via_is(self):
        """'col is null; DROP TABLE' should not parse."""
        with pytest.raises(FilterParseError, match="Unsupported filter syntax"):
            parse_filter("col is null; DROP TABLE users", allowed_cols={"col"})

    def test_in_list_with_injected_parens(self):
        """Nested parens in IN values become literal strings (quoted) or fail."""
        with pytest.raises(FilterParseError, match="Unquoted string literal"):
            parse_filter("col in (1, foo())", allowed_cols={"col"})


# ════════════════════════════════════════════════════════════════════════
# 6. any_filters (OR support)
# ════════════════════════════════════════════════════════════════════════

class TestAnyFilters:
    def test_or_only(self):
        sql, params = build_where(
            any_filters=["col = 'a'", "col = 'b'"],
            allowed_cols={"col"},
        )
        assert sql == 'WHERE ("col" = ? OR "col" = ?)'
        assert params == ["a", "b"]

    def test_and_plus_or(self):
        sql, params = build_where(
            filters=["split = 'train'"],
            any_filters=["speaker = 'jvs001'", "speaker = 'jvs002'"],
            allowed_cols={"split", "speaker"},
        )
        assert sql == (
            'WHERE "split" = ? AND ("speaker" = ? OR "speaker" = ?)'
        )
        assert params == ["train", "jvs001", "jvs002"]

    def test_and_only_unchanged(self):
        sql, params = build_where(
            filters=["col = 1", "col2 = 2"],
            allowed_cols={"col", "col2"},
        )
        assert sql == 'WHERE "col" = ? AND "col2" = ?'
        assert params == [1, 2]

    def test_empty_filters(self):
        sql, params = build_where()
        assert sql == ""
        assert params == []


# ════════════════════════════════════════════════════════════════════════
# 7. Schema-map canonicalization
# ════════════════════════════════════════════════════════════════════════

class TestSchemaMap:
    @pytest.fixture
    def smap(self):
        return schema_map_from_columns(["SpeakerID", "split", "Duration_s"])

    def test_builds_lowercase_keys(self, smap):
        assert set(smap.keys()) == {"speakerid", "split", "duration_s"}

    def test_canonicalize_case_insensitive(self, smap):
        assert canonicalize_column("SpeakerID", schema_map=smap) == "speakerid"
        assert canonicalize_column("SPLIT", schema_map=smap) == "split"
        assert canonicalize_column("duration_s", schema_map=smap) == "duration_s"

    def test_canonicalize_rejects_unknown(self, smap):
        with pytest.raises(FilterParseError, match="Unknown column"):
            canonicalize_column("nonexistent", schema_map=smap)

    def test_parse_filter_with_schema_map(self, smap):
        pf = parse_filter("split = 'train'", schema_map=smap)
        assert pf.sql == '"split" = ?'
        assert pf.params == ["train"]

    def test_build_where_with_schema_map(self, smap):
        sql, params = build_where(
            filters=["split = 'train'"],
            any_filters=["speakerid = 'jvs001'"],
            schema_map=smap,
        )
        assert '"split"' in sql
        assert '"speakerid"' in sql


# ════════════════════════════════════════════════════════════════════════
# 8. LIKE patterns (logical correctness, not injection)
# ════════════════════════════════════════════════════════════════════════

class TestLikePatterns:
    def test_like_wildcard_is_parameterized(self):
        pf = parse_filter("col LIKE '%'", allowed_cols={"col"})
        assert pf.sql == '"col" LIKE ?'
        assert pf.params == ["%"]

    def test_ilike(self):
        pf = parse_filter("col ilike '%test%'", allowed_cols={"col"})
        assert pf.sql == '"col" ILIKE ?'
        assert pf.params == ["%test%"]

    def test_like_rejects_non_string(self):
        with pytest.raises(FilterParseError, match="requires a string"):
            parse_filter("col LIKE 42", allowed_cols={"col"})


# ════════════════════════════════════════════════════════════════════════
# 9. NULL checks
# ════════════════════════════════════════════════════════════════════════

class TestNullChecks:
    def test_is_null(self):
        pf = parse_filter("col is null", allowed_cols={"col"})
        assert pf.sql == '"col" IS NULL'
        assert pf.params == []

    def test_is_not_null(self):
        pf = parse_filter("col is not null", allowed_cols={"col"})
        assert pf.sql == '"col" IS NOT NULL'
        assert pf.params == []


# ════════════════════════════════════════════════════════════════════════
# 10. parse_columns (column injection prevention)
# ════════════════════════════════════════════════════════════════════════

class TestParseColumns:
    def test_star_passthrough(self):
        assert parse_columns("*") == "*"

    def test_single_column(self):
        assert parse_columns("col", allowed_cols={"col"}) == '"col"'

    def test_multiple_columns(self):
        result = parse_columns("col1, col2", allowed_cols={"col1", "col2"})
        assert result == '"col1", "col2"'

    def test_rejects_subquery_injection(self):
        """The classic `-c "*, (select ...)"` attack."""
        with pytest.raises(FilterParseError):
            parse_columns("*, (select secret from passwords)", allowed_cols={"col"})

    def test_rejects_sql_comment_injection(self):
        with pytest.raises(FilterParseError):
            parse_columns("* FROM delta_scan('evil') --", allowed_cols={"col"})

    def test_rejects_unknown_column(self):
        with pytest.raises(FilterParseError, match="Unknown column"):
            parse_columns("secret", allowed_cols={"col"})

    def test_too_many_columns(self):
        cols = ", ".join([f"c{i}" for i in range(MAX_COLUMNS + 1)])
        allowed = {f"c{i}" for i in range(MAX_COLUMNS + 1)}
        with pytest.raises(FilterParseError, match="Too many columns"):
            parse_columns(cols, allowed_cols=allowed)

    def test_with_schema_map(self):
        smap = schema_map_from_columns(["SpeakerID", "split"])
        result = parse_columns("speakerid, split", schema_map=smap)
        assert result == '"speakerid", "split"'

    def test_empty_rejected(self):
        with pytest.raises(FilterParseError, match="Empty columns"):
            parse_columns("   ")


# ════════════════════════════════════════════════════════════════════════
# 11. safe_sql_string (string literal escaping)
# ════════════════════════════════════════════════════════════════════════

class TestSafeSqlString:
    def test_normal_path(self):
        assert safe_sql_string("/lake/silver/jsut/utterances") == "'/lake/silver/jsut/utterances'"

    def test_escapes_single_quotes(self):
        assert safe_sql_string("it's a path") == "'it''s a path'"

    def test_double_quote_breakout_attempt(self):
        """Attacker tries: path'); DROP TABLE --"""
        result = safe_sql_string("path'); DROP TABLE --")
        assert result == "'path''); DROP TABLE --'"
        # The doubled quote prevents breakout

    def test_rejects_null_byte(self):
        with pytest.raises(FilterParseError, match="null byte"):
            safe_sql_string("path\x00injection")

    def test_s3_path(self):
        result = safe_sql_string("s3://forge/lake/gold/koe/utterances")
        assert result == "'s3://forge/lake/gold/koe/utterances'"


# ════════════════════════════════════════════════════════════════════════
# 12. quote_spark_ident (Spark backtick quoting)
# ════════════════════════════════════════════════════════════════════════

class TestQuoteSparkIdent:
    def test_normal_ident(self):
        assert quote_spark_ident("gold_koe") == "`gold_koe`"

    def test_rejects_backtick(self):
        """Backtick rejected by regex before explicit check (both are defense-in-depth)."""
        with pytest.raises(FilterParseError, match="Invalid Spark identifier"):
            quote_spark_ident("gold`; DROP TABLE --")

    def test_rejects_special_chars(self):
        with pytest.raises(FilterParseError, match="Invalid Spark identifier"):
            quote_spark_ident("gold-koe")

    def test_rejects_leading_number(self):
        with pytest.raises(FilterParseError, match="Invalid Spark identifier"):
            quote_spark_ident("123gold")

    def test_allows_mixed_case(self):
        """Spark identifiers can be mixed case (unlike DuckDB SAFE_IDENT)."""
        assert quote_spark_ident("GoldKoe") == "`GoldKoe`"
