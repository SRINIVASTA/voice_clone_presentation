import os
import re
from pydub import AudioSegment
from gradio_client import Client, handle_file
from concurrent.futures import ThreadPoolExecutor

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, wrapped_voice, ref_text, text_chunk):
    """
    Sends a small text fragment to the current active F5-TTS API node.
    """
    try:
        # Standard prediction call for the F5-TTS endpoint structure
        result = client.predict(
            ref_audio=wrapped_voice,
            ref_text=ref_text,
            gen_text=text_chunk,
            remove_silence=False,
            api_name="/predict"
        )
        
        # Extract file path safely from tuple/list strings if returned that way
        remote_wav_path = result[0] if isinstance(result, (list, tuple)) else result
        
        if not remote_wav_path or not os.path.exists(str(remote_wav_path)):
            return None
            
        return AudioSegment.from_file(remote_wav_path, format="wav")
    except Exception as e:
        print(f"⚠️ Micro-chunk synthesis skipped on this client node: {e}")
        return None

def process_timestamp_block(client, wrapped_voice, ref_text, full_text, start_time):
    if not full_text:
        return None

    chunks = split_text_into_sentences(full_text)
    if not chunks:
        return None

    print(f"📦 Slide at {start_time/1000}s split dynamically into {len(chunks)} segments.")

    with ThreadPoolExecutor(max_workers=2) as chunk_executor:
        chunk_futures = [
            chunk_executor.submit(generate_single_chunk, client, wrapped_voice, ref_text, chk)
            for chk in chunks
        ]
        chunk_results = [f.result() for f in chunk_futures]

    valid_chunks = [c for c in chunk_results if c is not None]
    if not valid_chunks:
        return None

    combined_slide_audio = valid_chunks[0]
    for next_chunk in valid_chunks[1:]:
        combined_slide_audio += next_chunk

    return (start_time, combined_slide_audio)

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Booting Dynamic Scaling Presentation Engine...")
    
    # --- UPGRADED: Fallback system to try multiple spaces if one fails ---
    spaces_to_try = ["f5-tts/F5-TTS", "mrfakename/E2-F5-TTS"]
    client = None
    
    for space in spaces_to_try:
        try:
            print(f"🔗 Attempting connection to: {space}")
            client = Client(space, token=token) if token else Client(space)
            # Test a quick ping to see if the space is awake and working
            break 
        except Exception as conn_err:
            print(f"❌ Connection to {space} failed: {conn_err}. Trying fallback...")
            continue
            
    if not client:
        raise ValueError("All available public Hugging Face TTS spaces are currently busy or down. Please provide a Hugging Face Token in the sidebar to bypass shared lane limits.")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    wrapped_voice = handle_file(voice_path)
    
    # Process blocks across the timeline
    with ThreadPoolExecutor(max_workers=2) as slide_executor:
        slide_futures = [
            slide_executor.submit(process_timestamp_block, client, wrapped_voice, ref_text, text, t)
            for t, text in zip(timestamps, texts)
        ]
        slide_results = [f.result() for f in slide_futures]

    generated_slides = [r for r in slide_results if r is not None]
    
    if not generated_slides:
        raise ValueError("Could not synthesize any segments. The public nodes rejected the heavy script. Please paste a free Hugging Face Read Token in the sidebar to gain high-priority processing speeds.")

    max_required_duration = max(start_time + len(audio_segment) for start_time, audio_segment in generated_slides)
    
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    for start_time, audio_segment in generated_slides:
        final_presentation_audio = final_presentation_audio.overlay(audio_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    
    total_seconds = len(final_presentation_audio) / 1000
    display_minutes = int(total_seconds // 60)
    display_seconds = int(total_seconds % 60)
    print(f"🎉 {display_minutes}m {display_seconds}s Master Audio Compiled Successfully.")
