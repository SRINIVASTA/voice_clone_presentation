import os
import re
import io
import requests
from pydub import AudioSegment

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds):
    # Fetch your token directly from Streamlit Secrets
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Missing HF_TOKEN. Please add it to your Streamlit Secrets panel.")

    # API Endpoint for the official F5-TTS model on Hugging Face
    API_URL = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    generated_segments = []
    max_required_duration = 0

    # Read the voice sample once
    with open(voice_path, "rb") as f:
        voice_bytes = f.read()

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        # Send request to Hugging Face free Inference API
        payload = {
            "inputs": speech_text,
            "parameters": {
                "ref_audio": voice_bytes.hex(), # Transmit audio properties safely
                "ref_text": ref_text
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API Error: {response.text}")

        # Convert returned API audio data bytes into a Pydub segment
        speech_segment = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Build the silent canvas timeline
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    # Seamlessly overlay audio onto timestamps
    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
