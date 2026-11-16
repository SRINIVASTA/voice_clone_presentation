import re
from typing import List, Tuple


def parse_time_ms(time_str: str) -> int:
    """Converts [MM:SS] string into milliseconds."""
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp format: {time_str}")
    minutes, seconds = map(int, parts)
    if seconds >= 60 or minutes < 0 or seconds < 0:
        raise ValueError(f"Out-of-range timestamp values: {time_str}")
    return (minutes * 60 + seconds) * 1000


def parse_presentation_script(raw_text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Extracts optional Reference Text and sequential [MM:SS] speech segments."""
    if not raw_text or not raw_text.strip():
        return "", []

    ref_match = re.search(r"(?:Reference Text:\s*)(.*)", raw_text, re.IGNORECASE)
    cleaned_body = re.sub(r"(?:Reference Text:\s*)(.*)", "", raw_text, flags=re.IGNORECASE)

    segments = re.split(r"\[(\d{2}:\d{2})\]", cleaned_body)[1:]
    timeline_items: List[Tuple[int, str]] = []

    for i in range(0, len(segments), 2):
        timestamp_ms = parse_time_ms(segments[i])
        speech_text = segments[i + 1].strip()
        if speech_text:
            timeline_items.append((timestamp_ms, speech_text))

    timeline_items.sort(key=lambda item: item[0])

    if ref_match and ref_match.group(1).strip():
        ref_text = ref_match.group(1).strip()
    elif timeline_items:
        ref_text = timeline_items[0][1]
    else:
        ref_text = ""

    return ref_text, timeline_items
