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
    Directly routes structured payloads via clean HTTP POST calls to the F5-TTS container.
    Bypasses text_to_speech client wrappers to prevent infinite buffering parameters.
    """
    # Using the optimized serverless router lane for stable processing
    url = "https://huggingface.co"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json"
    }
    
    # 🌟 FIXED CRITICAL PAYLOAD: F5-TTS architecture matches this exact schema layout
    payload = {
        "inputs": str(text_chunk).strip(),
        "parameters": {
            "reference_audio": str(voice_bytes_b64),
            "reference_text": str(ref_text).strip()
        }
    }
    
    # Basic single handshake block to break infinite buffering on server congestion
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        
        # Explicit status checks instead of hanging loop paths
        if response.status_code == 503:
            print("⏳ Model is currently spinning up on serverless nodes. Waiting 10s...")
            time.sleep(10)
            # Single backup handshake query
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            
        if response.status_code == 200 and len(response.content) > 1000:
            return AudioSegment.from_file(io.BytesIO(response.content))
            
        print(f"⚠️ Server returned unhandled code ({response.status_code}) - Data chunk skipped.")
    except Exception as e:
        print(f"⚠️ Gateway transmission exception error: {e}")
        
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Synchronizing timeline data tracks across cluster networks...")
    
    if not token or not token.strip():
        raise ValueError("A fully privileged Hugging Face Access Token is required to complete generation.")

    # --- NORMALIZE SPEAKER PROFILE ---
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        # Force a lean 8-second file snippet footprint block to guarantee fast transfers
        optimized_voice = raw_voice[:8000] 
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as audio_err:
        raise ValueError(f"Failed to cleanly normalize input speaker profile track: {audio_err}")

    # --- DECODE PRESENTATION SCRIPTS ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Timeline structure is invalid. Please format scripts with explicit [MM:SS] timestamps.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- LAYOUT SEGMENT 1: INJECT THE REFERENCE OVERLAY AS INTRO SLIDE 1 ---
    print("🎤 Directing baseline intro track...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, chunk, token)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=200)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Slide transition break gap
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- LAYOUT SEGMENT 2: MERGE REMAINING TIMELINE PARAGRAPHS ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Swap acronym terms to guarantee smooth vocal execution profiles
            clean_chunk = chunk.replace("ML", "machine learning").replace("UI", "user interface").replace("JS", "javascript").replace("PDF", "document report")
            
            sentence_audio = generate_single_chunk(voice_bytes_b64, ref_text, clean_chunk, token)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)

        # Build precise chronological silence pacing arrays onto the master layout grid canvas
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Halt file creation if public node overloads returned empty segments
    if len(final_presentation_audio) <= intro_duration + 500:
        raise ValueError("The public server clusters are completely full. No audio data was generated. Please wait 1 minute and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Output Save
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Total Presentation Timeline Compiled Successfully.")
