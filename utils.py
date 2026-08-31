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
    Sends a text fragment to the stable serverless model engine.
    Fixed: Uses the proper nested input structure and robust error fallbacks.
    """
    try:
        # 🌟 FIXED SCHEMA: F5-TTS requires an explicitly nested input dictionary block
        payload = {
            "inputs": {
                "text": str(text_chunk).strip(),
                "reference_audio": str(voice_bytes_b64),
                "reference_text": str(ref_text).strip()
            }
        }

        # Query the model endpoint directly
        response_bytes = client.post(
            model="m-a-p/F5-TTS",
            json=payload
        )
        
        # Guard against zero-byte empty returns or broken handshakes
        if not response_bytes or len(response_bytes) < 500:
            print(f"⚠️ Empty payload or silence returned from backend for text: {text_chunk}")
            return None
            
        # Convert raw binary array response straight to an AudioSegment
        return AudioSegment.from_file(io.BytesIO(response_bytes))
    except Exception as e:
        print(f"⚠️ Web API processing segment error: {e}")
        return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting to the official Hugging Face serverless engine...")
    
    if not token or not token.strip():
        raise ValueError("A Hugging Face Token is required to authenticate with the serverless API. Please paste your token in the sidebar password box.")

    try:
        client = InferenceClient(token=token.strip())
    except Exception as e:
        raise ValueError(f"Failed to initialize the Hugging Face client: {e}")

    # --- 🛠️ ROBUST AUDIO PAYLOAD NORMALIZATION ---
    try:
        print("⚡ Optimizing reference audio profile...")
        raw_voice = AudioSegment.from_file(voice_path)
        
        # Trims the first 15 seconds to drop payload size down dramatically
        optimized_voice = raw_voice[:15000] 
        
        # Enforce exact standard 24000Hz sampling rate and 16-bit PCM channels
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        # Export with clean, minimalist parameter headers to avoid confusing the cloud tensor inputs
        buffer = io.BytesIO()
        optimized_voice.export(buffer, format="wav", codec="pcm_s16le")
        voice_bytes_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("✅ Voice optimized and converted to payload cleanly.")
    except Exception as audio_err:
        raise ValueError(f"Failed to process or read reference audio file: {audio_err}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract valid timestamps from your script. Ensure format matches [MM:SS].")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            sentence_audio = generate_single_chunk(client, voice_bytes_b64, ref_text, chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=250) # Natural pacing gap

        # Handle chronological layout mapping
        if start_time > current_timeline_position:
            silence_needed = start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Throw explicit runtime warning instead of rendering a broken silent track file
    if len(final_presentation_audio) == 0:
        raise ValueError("The compiled audio timeline is completely empty. The Hugging Face serverless endpoint might be busy. Please try again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Export cleanly to local workspace disk
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 Compiled Successfully via Serverless API.")
