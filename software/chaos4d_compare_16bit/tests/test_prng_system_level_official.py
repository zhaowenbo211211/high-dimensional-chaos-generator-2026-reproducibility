# Maintainer: zhaowenbo
from metrics.prng_system_level_official import (
    OFFICIAL_STS_CATEGORY_ORDER,
    build_system_level_schemes,
    generate_system_level_payload,
    parse_official_summary_rows_by_category,
)


def test_generate_system_level_payload_sizes() -> None:
    schemes = build_system_level_schemes()
    for scheme in schemes:
        payload = generate_system_level_payload(
            scheme=scheme,
            sequence_bit_length=1024,
            sequence_count=2,
            warmup_steps=1000,
        )
        assert len(payload) == 256


def test_parse_official_summary_rows_by_category_collapses_subtests() -> None:
    rows = [
        {"test_name": "Frequency", "raw_line": " 10 10 10 10 10 10 10 10 10 10  0.500000    99/100     Frequency"},
        {"test_name": "NonOverlappingTemplate", "raw_line": " 10 10 10 10 10 10 10 10 10 10  0.500000    99/100     NonOverlappingTemplate"},
        {"test_name": "NonOverlappingTemplate", "raw_line": " 10 10 10 10 10 10 10 10 10 10  0.000000 *   90/100  *  NonOverlappingTemplate"},
    ]
    parsed = parse_official_summary_rows_by_category(rows)
    assert [row["test_name"] for row in parsed] == list(OFFICIAL_STS_CATEGORY_ORDER)
    frequency = next(row for row in parsed if row["test_name"] == "Frequency")
    non_overlapping = next(row for row in parsed if row["test_name"] == "NonOverlapping Template Matchings")
    assert bool(frequency["passed"]) is True
    assert bool(non_overlapping["passed"]) is False
    assert int(non_overlapping["flagged_subtests"]) == 1
