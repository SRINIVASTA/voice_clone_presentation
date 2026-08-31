import io
import os
import re
import time
import base64
import requests
from pydub import AudioSegment

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(voice_bytes_b64, ref_text, text_chunk, token):
    """
    Queries the core serverless F5-TTS endpoint with automated retry handling.
    Bypasses community Gradio spaces entirely to prevent 404 repository issues.
    """
    url = "https://huggingface.co"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json"
    }
    
    # 🌟 CRUCIAL API MAPPING SCHEMA FOR SERVERLESS INFRASTRUCTURE
    payload = {
        "inputs": str(text_chunk).strip(),
        "parameters": {
            "reference_audio": str(voice_bytes_b64),
            "reference_text": str(ref_text).strip()
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Handle 503 Model Loading status and allow server node to warm up weights
            if response.status_code == 503:
                time.sleep(15)
                continue
                
            # Handle 504 HTTP Gateway timeouts smoothly
            if response.status_code == 504:
                time.sleep(5)
                continue
                
            # If successful and returns true raw waveform binary, pass it down to pydub
            if response.status_code == 200 and len(response.content) > 1000:
                return AudioSegment.from_file(io.BytesIO(response.content))
                
            print(f"⚠️ Server returned unhandled response code ({response.status_code}) for text segment.")
        except Exception as e:
            time.sleep(3)
            
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Handshaking with Hugging Face Serverless Core Cluster...")
    
    if not token or not token.strip():
        raise ValueError("A valid Hugging Face User Access Token is required. Please paste it into the sidebar panel.")

    # --- VOICE BLUEPRINT PROFILING ---
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        # Constrain voice tracking snippet data block size to avoid hitting API buffer caps
        optimized_voice = raw_voice[:10000] # Safe 10-second vocal blueprint
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as audio_err:
        raise ValueError(f"Failed to process your reference sound track: {audio_err}")

    # --- READ TIMELINE SCRIPT FILE ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Timeline structure is invalid. Please format scripts with explicit [MM:SS] timestamps.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: INJECT THE REFERENCE TEXT BLOCK AS INTRO SLIDE 1 ---
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, chunk, token)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=200)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Structural slide buffer gap
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- SECTION 2: COMPILE THE REMAINING PRESENTATION LINES ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Expand common developer acronyms so the neural layer speaks them clearly as complete words
            clean_chunk = chunk.replace("ML", "machine learning").replace("UI", "user interface").replace("JS", "javascript").replace("PDF", "document report")
            
            sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, clean_chunk, token)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)

        # Map timeline spacing dynamically onto the timeline audio grid canvas
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Raise clear error if global cluster points returned no vocal outputs
    if len(final_presentation_audio) <= intro_duration + 500:
        raise ValueError("The public serverless clusters are heavily congested right now. Please wait a moment and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Output Save
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Total Presentation Timeline Compiled Successfully.")
