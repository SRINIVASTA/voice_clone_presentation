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
    Queries the F5-TTS model over a robust payload schema with automatic retries.
    """
    url = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {token.strip()}"}
    
    # 🌟 FIXED SCHEMA KEYS: Exact dictionary key match for F5-TTS Hugging Face Endpoint
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
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            
            # Catch 503 Model Loading status and wait for warmup
            if response.status_code == 503:
                print(f"⏳ Hugging Face model warming up... Waiting 15s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(15)
                continue
                
            # Handle 504 Gateway Timeouts
            if response.status_code == 504 or b"Gateway Timeout" in response.content:
                print(f"⚠️ Gateway Timeout encountered. Retrying in 5s...")
                time.sleep(5)
                continue
                
            # If successful, return the valid audio segment block
            if response.status_code == 200 and len(response.content) > 500:
                return AudioSegment.from_file(io.BytesIO(response.content))
                
            print(f"⚠️ Server returned unhandled status code ({response.status_code}) for fragment: {text_chunk}")
            
        except Exception as e:
            print(f"⚠️ Connection anomaly on attempt {attempt + 1}: {e}")
            time.sleep(3)
            
    print(f"❌ Failed to render phrase segment: '{text_chunk}'")
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting to Hugging Face cluster nodes...")
    
    if not token or not token.strip():
        raise ValueError("A Hugging Face Token is required to authenticate. Please paste your token in the sidebar dashboard box.")

    # --- VOICE PRE-PROCESSING AND STRUCTURAL NORMALIZATION ---
    try:
        print("⚡ Resampling and trimming reference voice file...")
        raw_voice = AudioSegment.from_file(voice_path)
        
        # Limit voice processing to the first 12 seconds to keep network payloads small
        optimized_voice = raw_voice[:12000] 
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("✅ Voice optimized and converted cleanly.")
    except Exception as audio_err:
        raise ValueError(f"Failed to process your reference audio file: {audio_err}")

    # --- PARSE PRESENTATION FILE TIMELINES ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract clean timestamps from your presentation file. Ensure it follows [MM:SS] rules.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: INJECTING REFERENCE TEXT AS INTRO SLIDE ---
    print("🎤 Compiling Reference Text as Slide 1 Intro...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, chunk, token)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=250)
            
    final_presentation_audio += AudioSegment.silent(duration=1500)  # Slide transition delay spacing
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration
    print(f"✅ Slide 1 Intro generation step complete.")

    # --- SECTION 2: COMPILING MAIN PRESENTATION SLIDES ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        # Adjust the upcoming timeline blocks forward to fit nicely right after the intro slide
        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, chunk, token)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=250)

        # Handle chronological layout stitching
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Safety Guard: Stop execution if everything returned silent arrays
    if len(final_presentation_audio) <= intro_duration + 1000:
        raise ValueError("The Hugging Face endpoint returned empty data arrays. The public container nodes are heavily busy. Please wait a few moments and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Export
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Total Presentation Timeline Compiled Successfully.")
