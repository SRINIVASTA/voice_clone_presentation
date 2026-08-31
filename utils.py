import io
import os
import re
import time
import base64
from pydub import AudioSegment
from huggingface_hub import InferenceClient

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, voice_bytes_b64, ref_text, text_chunk):
    """
    Synthesizes voice chunks natively through the standard Inference Client text-to-speech mapping.
    This schema guarantees proper internal dictionary formatting over the cloud gateway.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 🌟 NATIVE INFERENCE: Calls text_to_speech directly to manage payload construction automatically
            # Targeted directly at the official base model path
            response_bytes = client.text_to_speech(
                text=str(text_chunk).strip(),
                model="m-a-p/F5-TTS",
                parameters={
                    "reference_audio": str(voice_bytes_b64),
                    "reference_text": str(ref_text).strip()
                }
            )
            
            # Catch raw waveform audio binary back cleanly
            if response_bytes and len(response_bytes) > 1000:
                return AudioSegment.from_file(io.BytesIO(response_bytes))
                
        except Exception as e:
            # Detect model warming up errors (503) or gateway overloads (504)
            err_msg = str(e)
            if "503" in err_msg or "loading" in err_msg.lower():
                print(f"⏳ Model is currently loading on Hugging Face nodes... Waiting 15s...")
                time.sleep(15)
                continue
            elif "504" in err_msg:
                print(f"⏳ Gateway timeout encountered. Retrying segment...")
                time.sleep(5)
                continue
            else:
                print(f"⚠️ Chunk compilation warning: {e}")
                time.sleep(3)
                
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting via Hugging Face Inference Client Suite...")
    
    if not token or not token.strip():
        raise ValueError("A valid Hugging Face User Access Token is required to authenticate. Please check the sidebar box.")

    try:
        # Initialize official inference gateway engine handler
        client = InferenceClient(token=token.strip())
    except Exception as e:
        raise ValueError(f"Failed to securely initialize the Hugging Face client: {e}")

    # --- REFERENCE SOUND PROFILING CONSTRAINTS ---
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        # Downsample file size footprints to prevent 504 proxy transfer cuts
        optimized_voice = raw_voice[:10000] # Clean 10s tracking footprint
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as audio_err:
        raise ValueError(f"Failed to normalize input vocal blueprint: {audio_err}")

    # --- READ TIMELINE PRESENTATION TEXT FILE ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Timeline structure is invalid. Please format scripts with explicit [MM:SS] timestamps.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: COMPILING THE INTRO TRACK AS SLIDE 1 ---
    print("🎤 Generating Slide 1 Intro narration baseline...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(client, voice_bytes_b64, ref_text, chunk)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=200)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Structural slide buffer gap
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- SECTION 2: PROCESSING THE REMAINING TIMELINE SEGMENTS ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Substitute quick shortcuts on the fly so the acoustic voice handles pronunciation smoothly
            clean_chunk = chunk.replace("ML", "machine learning").replace("UI", "user interface").replace("JS", "javascript").replace("PDF", "document report")
            
            sentence_audio = generate_single_chunk(client, voice_bytes_b64, ref_text, clean_chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)

        # Build chronological spacing dynamically onto the timeline audio grid canvas
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Halt compilation if everything processed as perfect silence due to network drops
    if len(final_presentation_audio) <= intro_duration + 500:
        raise ValueError("The public serverless inference nodes are currently overloaded. Please wait 1 minute and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Output Save
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Total Presentation Timeline Compiled Successfully.")
