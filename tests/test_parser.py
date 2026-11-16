import pytest
from src.parser import parse_time_ms, parse_presentation_script


def test_parse_time_ms_valid():
    assert parse_time_ms("00:00") == 0
    assert parse_time_ms("01:30") == 90000


def test_parse_time_ms_invalid():
    with pytest.raises(ValueError):
        parse_time_ms("01:75")


def test_parse_presentation_script_with_ref():
    sample = (
        "Reference Text: Hello team, welcome.\n"
        "[00:00] First slide intro.\n"
        "[00:15] Second slide metrics."
    )
    ref, items = parse_presentation_script(sample)
    assert ref == "Hello team, welcome."
    assert len(items) == 2
    assert items[0] == (0, "First slide intro.")
    assert items[1] == (15000, "Second slide metrics.")


def test_parse_presentation_script_fallback_ref():
    sample = "[00:05] Opening remark.\n[00:20] Closing remark."
    ref, items = parse_presentation_script(sample)
    assert ref == "Opening remark."
    assert len(items) == 2
