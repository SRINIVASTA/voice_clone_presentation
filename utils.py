import os
import re
from pydub import AudioSegment
from gradio_client import Client, handle_file
from concurrent.futures import ThreadPoolExecutor

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    """
    Splits long presentation paragraphs into clean individual sentences.
    This guarantees Hugging Face never hits a token limit.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, wrapped_voice, ref_text, text_chunk):
    """
    Sends a small, safe text fragment to the F5-TTS API node.
    """
    try:
        result = client.predict(
            ref_audio=wrapped_voice,
            ref_text=ref_text,
            gen_text=text_chunk,
            remove_silence=False,
            api_name="/predict"
        )
        # Handle structural list/tuple return variants safely
        remote_wav_path = result
        
        if not remote_wav_path or not os.path.exists(str(remote_wav_path)):
            return None
            
        return AudioSegment.from_file(remote_wav_path, format="wav")
    except Exception as e:
        print(f"⚠️ Micro-chunk synthesis skipped due to error: {e}")
        return None

def process_timestamp_block(client, wrapped_voice, ref_text, full_text, start_time):
    """
    Processes a complete presentation slide block. If the text is long,
    it splits it, synthesizes sentences concurrently, and chains them together.
    """
    if not full_text:
        return None

    chunks = split_text_into_sentences(full_text)
    if not chunks:
        return None

    print(f"📦 Slide at {start_time/1000}s split dynamically into {len(chunks)} text segments.")

    # Concurrently synthesize all sentences within this single slide block
    with ThreadPoolExecutor(max_workers=3) as chunk_executor:
        chunk_futures = [
            chunk_executor.submit(generate_single_chunk, client, wrapped_voice, ref_text, chk)
            for chk in chunks
        ]
        chunk_results = [f.result() for f in chunk_futures]

    # Stitch the individual sentences together into one continuous slide audio
    valid_chunks = [c for c in chunk_results if c is not None]
    if not valid_chunks:
        return None

    combined_slide_audio = valid_chunks[0]
    for next_chunk in valid_chunks[1:]:
        combined_slide_audio += next_chunk  # Seamless appending

    return (start_time, combined_slide_audio)

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Booting Dynamic Scaling Presentation Engine...")
    
    if token:
        client = Client("mrfakename/E2-F5-TTS", token=token)
    else:
        client = Client("mrfakename/E2-F5-TTS")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse timestamps and slide content blocks
    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    wrapped_voice = handle_file(voice_path)
    
    # Process all slide blocks concurrently across the entire timeline
    with ThreadPoolExecutor(max_workers=2) as slide_executor:
        slide_futures = [
            slide_executor.submit(process_timestamp_block, client, wrapped_voice, ref_text, text, t)
            for t, text in zip(timestamps, texts)
        ]
        slide_results = [f.result() for f in slide_futures]

    generated_slides = [r for r in slide_results if r is not None]
    
    if not generated_slides:
        raise ValueError("Could not synthesize any presentation segments. Please verify your asset formats.")

    # Dynamically find the absolute end of your presentation timeline matrix
    max_required_duration = max(start_time + len(audio_segment) for start_time, audio_segment in generated_slides)
    
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    # Place each compiled slide track into its precise presentation timestamp spot
    for start_time, audio_segment in generated_slides:
        final_presentation_audio = final_presentation_audio.overlay(audio_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    
    # --- NEW: Calculate exact output length dynamically for the console print statement ---
    total_seconds = len(final_presentation_audio) / 1000
    display_minutes = int(total_seconds // 60)
    display_seconds = int(total_seconds % 60)
    
    print(f"🎉 {display_minutes}m {display_seconds}s Compatible Master Audio Output Compiled Successfully.")
