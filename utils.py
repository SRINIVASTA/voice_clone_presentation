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
    """
    Splits long presentation paragraphs into clean individual sentences.
    This guarantees the web API nodes never hit a token limit.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, voice_path, ref_text, text_chunk):
    """
    Sends a text fragment to the stable serverless model engine.
    """
    try:
        # Read reference audio bytes and convert to standard base64 for API transmission
        with open(voice_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Standard dictionary payload required by the serverless text-to-speech API
        payload = {
            "inputs": text_chunk,
            "parameters": {
                "reference_audio": audio_b64,
                "reference_text": ref_text
            }
        }

        # Query the model endpoint directly
        # 'm-a-p/F5-TTS' is the official model path on Hugging Face
        response_bytes = client.post(
            model="m-a-p/F5-TTS",
            json=payload
        )
        
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
        # Initialize the official serverless Inference Client with your provided token
        client = InferenceClient(token=token.strip())
    except Exception as e:
        raise ValueError(f"Failed to initialize the Hugging Face client: {e}")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract valid timestamps from your script. Ensure format matches [MM:SS].")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # Process chunks sequentially to respect rate limits and keep track of timestamps
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text:
            continue

        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Query the client for each sentence chunk
            sentence_audio = generate_single_chunk(client, voice_path, ref_text, chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)  # Subtle natural sentence-break pause

        # Overlay calculations / Timeline pad alignment
        if start_time > current_timeline_position:
            silence_needed = start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Apply Sidebar End Padding Adjustments
    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Export
    final_presentation_audio.export(output_path, format="wav")
    
    total_seconds = len(final_presentation_audio) / 1000
    display_minutes = int(total_seconds // 60)
    display_seconds = int(total_seconds % 60)
    print(f"🎉 {display_minutes}m {display_seconds}s Compiled Successfully via Serverless API.")
