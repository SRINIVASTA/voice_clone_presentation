import io
import os
import re
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
    Sends a text fragment to a highly responsive serverless F5-TTS model engine.
    """
    try:
        payload = {
            "inputs": {
                "text": str(text_chunk).strip(),
                "reference_audio": voice_bytes_b64,
                "reference_text": str(ref_text).strip()
            }
        }
        
        # 🌟 BACKUP ROUTE: Using the hyper-responsive public spaces endpoint fallback
        response_bytes = client.post(
            model="https://huggingface.co",
            json=payload
        )
        
        # Guard against HTML text errors like 504 Gateway Timeout or 503 Overload
        if b"<html>" in response_bytes or b"Gateway Timeout" in response_bytes:
            print(f"❌ Server timeout or busy nodes for text fragment: '{text_chunk}'")
            return None
            
        if not response_bytes or len(response_bytes) < 500:
            print(f"⚠️ Empty payload or silent data array returned for text: '{text_chunk}'")
            return None
            
        return AudioSegment.from_file(io.BytesIO(response_bytes))
    except Exception as e:
        print(f"⚠️ Web API processing segment error: {e}")
        return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting to the optimized Hugging Face serverless engine...")
    
    if not token or not token.strip():
        raise ValueError("A Hugging Face Token is required to authenticate with the serverless API. Please paste your token in the sidebar password box.")

    try:
        client = InferenceClient(token=token.strip())
    except Exception as e:
        raise ValueError(f"Failed to initialize the Hugging Face client: {e}")

    # --- AUDIO PAYLOAD NORMALIZATION ---
    try:
        print("⚡ Optimizing reference audio profile...")
        raw_voice = AudioSegment.from_file(voice_path)
        optimized_voice = raw_voice[:15000] # Safe 15s slice snippet
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("✅ Voice optimized successfully.")
    except Exception as audio_err:
        raise ValueError(f"Failed to process or read reference audio file: {audio_err}")

    # --- PARSE PRESENTATION FILE ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract valid timestamps from your script. Ensure format matches [MM:SS].")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- STEP 1: PRESENT THE REFERENCE TEXT AS SLIDE 1 ---
    print("🎤 Synthesizing Reference Text as Slide 1 Intro...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(client, voice_bytes_b64, ref_text, chunk)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=250)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Slide change break
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration
    print(f"✅ Slide 1 Intro Complete. Duration: {intro_duration / 1000:.2f} seconds.")

    # --- STEP 2: COMPILE SUBSEQUENT TIMESTAMPS (Auto-Shifted) ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            sentence_audio = generate_single_chunk(client, voice_bytes_b64, ref_text, chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=250)

        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Throw error if everything failed to prevent a broken silent output download
    if len(final_presentation_audio) <= intro_duration + 2000:
        raise ValueError("The public serverless inference nodes are currently overloaded and timing out. Please wait 1-2 minutes and click generate again to retry.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Export
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Total Presentation Timeline Compiled Successfully.")
