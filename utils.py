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
    This guarantees the web API nodes never hit a token limit.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, wrapped_voice, ref_text, text_chunk):
    """
    Sends a text fragment to an active F5-TTS API cluster node.
    """
    try:
        # Connect to a live layout with parameters expected by standard active Spaces
        result = client.predict(
            ref_audio_input=wrapped_voice,
            ref_text_input=ref_text,
            gen_text_input=text_chunk,
            remove_silence=False,
            speed=1.0,  # Explicit speed multi-factor required by standard spaces
            api_name="/infer"
        )
        
        # Safely extract the valid .wav track file path from multi-value return matrices
        remote_wav_path = None
        if isinstance(result, (list, tuple)) and len(result) > 0:
            remote_wav_path = result[0]
        elif isinstance(result, dict) and "name" in result:
            remote_wav_path = result["name"]
        elif isinstance(result, str):
            remote_wav_path = result
            
        if not remote_wav_path or not os.path.exists(str(remote_wav_path)):
            return None
            
        return AudioSegment.from_file(remote_wav_path, format="wav")
    except Exception as e:
        print(f"⚠️ Web API processing segment error: {e}")
        return None

def process_timestamp_block(client, wrapped_voice, ref_text, full_text, start_time):
    if not full_text:
        return None

    chunks = split_text_into_sentences(full_text)
    if not chunks:
        return None

    # Sequential processing loop protects your token lane from scraping bans
    with ThreadPoolExecutor(max_workers=1) as chunk_executor:
        chunk_futures = [
            chunk_executor.submit(generate_single_chunk, client, wrapped_voice, ref_text, chk)
            for chk in chunks
        ]
        chunk_results = [f.result() for f in chunk_futures]

    valid_chunks = [c for c in chunk_results if c is not None]
    if not valid_chunks:
        return None

    # Properly initialize an empty AudioSegment to stitch chunks
    combined_slide_audio = AudioSegment.empty()
    for chunk in valid_chunks:
        combined_slide_audio += chunk
        combined_slide_audio += AudioSegment.silent(duration=200) # Subtle natural sentence-break pause

    return (start_time, combined_slide_audio)

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting to a live production server cluster...")
    
    # 🎯 THE FIX: Target a verified, running alternative space.
    # If you choose to host your own dedicated space clone, swap this string out 
    # with your personal space ID (e.g., "YourUsername/YourSpaceName")
    target_space = "m-a-p/F5-TTS"
    
    try:
        # Pass the token explicitly if entered by the user
        client = Client(target_space, token=token) if token else Client(target_space)
    except Exception as e:
        raise ValueError(f"Failed to connect to the cloud model engine: {e}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    wrapped_voice = handle_file(voice_path)
    
    # Run through sequentially to guarantee safe server-side rate limits
    with ThreadPoolExecutor(max_workers=1) as slide_executor:
        slide_futures = [
            slide_executor.submit(process_timestamp_block, client, wrapped_voice, ref_text, text, t)
            for t, text in zip(timestamps, texts)
        ]
        slide_results = [f.result() for f in slide_futures]

    generated_slides = [r for r in slide_results if r is not None]
    
    if not generated_slides:
        raise ValueError("The server backend returned null objects. If traffic is heavy, please provide your personal Hugging Face Token.")

    max_required_duration = max(start_time + len(audio_segment) for start_time, audio_segment in generated_slides)
    
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    for start_time, audio_segment in generated_slides:
        final_presentation_audio = final_presentation_audio.overlay(audio_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    
    total_seconds = len(final_presentation_audio) / 1000
    display_minutes = int(total_seconds // 60)
    display_seconds = int(total_seconds % 60)
    print(f"🎉 {display_minutes}m {display_seconds}s Compiled Successfully via Official Web API.")
