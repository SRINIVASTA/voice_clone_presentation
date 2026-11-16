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


def _split_into_short_sentences(text: str, max_chars: int = 140) -> List[str]:
    """Splits long paragraphs into natural sentence chunks so F5-TTS never drops words."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split long sentences on commas if needed
            sub_parts = [p.strip() for p in sentence.split(",") if p.strip()]
            chunks.extend(sub_parts)
    return chunks


def parse_presentation_script(raw_text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Extracts Reference Text and splits [MM:SS] blocks into short, clean speech chunks."""
    if not raw_text or not raw_text.strip():
        return "", []

    ref_match = re.search(r"(?:Reference Text:\s*)(.*)", raw_text, re.IGNORECASE)
    cleaned_body = re.sub(r"(?:Reference Text:\s*)(.*)", "", raw_text, flags=re.IGNORECASE)

    segments = re.split(r"\[(\d{2}:\d{2})\]", cleaned_body)[1:]
    timeline_items: List[Tuple[int, str]] = []

    for i in range(0, len(segments), 2):
        timestamp_ms = parse_time_ms(segments[i])
        speech_block = segments[i + 1].strip()
        if not speech_block:
            continue

        short_chunks = _split_into_short_sentences(speech_block)
        for idx, chunk in enumerate(short_chunks):
            # First chunk anchors at the exact [MM:SS] timestamp; subsequent chunks follow naturally
            chunk_ts = timestamp_ms if idx == 0 else 0
            timeline_items.append((chunk_ts, chunk))

    if ref_match and ref_match.group(1).strip():
        ref_text = ref_match.group(1).strip()
    elif timeline_items:
        ref_text = timeline_items
    else:
        ref_text = ""

    return ref_text, timeline_items
