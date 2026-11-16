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


def _split_into_clean_sentences(text: str, max_chars: int = 160) -> List[str]:
    """Splits paragraphs on sentence endings while avoiding tiny 2-3 word fragments."""
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            # Merge very short sentences (< 25 chars) into the previous chunk if possible
            if chunks and len(sentence) < 25 and (len(chunks[-1]) + len(sentence) + 1) <= max_chars:
                chunks[-1] = f"{chunks[-1]} {sentence}"
            else:
                chunks.append(sentence)
        else:
            # Only split long sentences at mid-point commas so fragments stay substantial
            midpoint = len(sentence) // 2
            comma_positions = [m.start() for m in re.finditer(r",", sentence)]
            if comma_positions:
                best_comma = min(comma_positions, key=lambda pos: abs(pos - midpoint))
                part1 = sentence[: best_comma + 1].strip()
                part2 = sentence[best_comma + 1 :].strip()
                if part1:
                    chunks.append(part1)
                if part2:
                    chunks.append(part2)
            else:
                chunks.append(sentence)

    return chunks


def parse_presentation_script(raw_text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Extracts Reference Text string and splits [MM:SS] blocks into clean speech chunks."""
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

        short_chunks = _split_into_clean_sentences(speech_block)
        for idx, chunk in enumerate(short_chunks):
            chunk_ts = timestamp_ms if idx == 0 else 0
            timeline_items.append((chunk_ts, chunk))

    if ref_match and ref_match.group(1).strip():
        ref_text = ref_match.group(1).strip()
    elif timeline_items:
        # Strictly extract ONLY the first sentence string, never the whole list
        ref_text = timeline_items[0][1]
    else:
        ref_text = ""

    return ref_text, timeline_items
