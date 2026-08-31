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
    Sends presentation sentences directly to the F5-TTS serverless instance.
    Uses the exact expected payload formatting required by Hugging Face.
    """
    url = "https://huggingface.co"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json"
    }
    
    # 🌟 CRUCIAL API MAPPING LAYOUT
    payload = {
        "inputs": text_chunk,
        "parameters": {
            "reference_audio": voice_bytes_b64,
            "reference_text": ref_text
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # If the model is resting, wait for the server to load weights
            if response.status_code == 503:
                time.sleep(15)
                continue
                
            # Handle server congestions/timeouts 
            if response.status_code == 504:
                time.sleep(5)
                continue
                
            # If it returns a valid audio file size stream, send it directly to the timeline stitcher
            if response.status_code == 200 and len(response.content) > 1000:
                return AudioSegment.from_file(io.BytesIO(response.content))
                
            print(f"⚠️ Server Response Code ({response.status_code}) - Retrying chunk...")
        except Exception as e:
            time.sleep(3)
            
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting to Hugging Face Cloud Infrastructure...")
    
    if not token or not token.strip():
        raise ValueError("Please provide a valid Hugging Face Access Token in the app sidebar text box.")

    # --- ENFORCE DENSE AUDIO PROFILING ---
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        # Squeeze down the file size to prevent 504 Gateway HTTP timeouts
        optimized_voice = raw_voice[:10000] # Safe 10-second reference footprint
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as audio_err:
        raise ValueError(f"Failed to optimize input voice sample file: {audio_err}")

    # --- READ SCRIPTS ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract timelines. Script must follow [MM:SS] styling rules.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: INJECT THE REFERENCE INTRO TRACK AS SLIDE 1 ---
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, chunk, token)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=200)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Structural slide buffer gap
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- SECTION 2: PROCESS THE REST OF THE TIMELINES ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Swap acronyms so the TTS model engine doesn't hit processing glitches
            clean_chunk = chunk.replace("ML", "machine learning").replace("UI", "user interface").replace("JS", "javascript").replace("PDF", "document report")
            
            sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, clean_chunk, token)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)

        # Build chronological spacing dynamically
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Stop execution if public endpoints fail to return content
    if len(final_presentation_audio) <= intro_duration + 500:
        raise ValueError("The public server clusters are completely full. No audio data was generated. Please wait 1 minute and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Output Save
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
