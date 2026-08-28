import os
import re
import requests
import io
from pydub import AudioSegment

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds):
    # Ensure you have your free token set up in Streamlit Secrets
    REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
    if not REPLICATE_API_TOKEN:
        raise ValueError("Missing REPLICATE_API_TOKEN. Please set it up in your Streamlit dashboard secrets.")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    generated_segments = []
    max_required_duration = 0

    # Prepare your voice clip to send to the API
    with open(voice_path, "rb") as voice_file:
        voice_bytes = voice_file.read()

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        # 🚀 Send voice generation block to a remote cloud GPU API (e.g., F5-TTS on Replicate)
        headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
        payload = {
            "version": "lucataco/f5-tts:de21cf81", # Example public F5-TTS container
            "input": {
                "gen_text": speech_text,
                "ref_text": ref_text,
                "ref_audio": f"data:audio/wav;base64,{voice_bytes.hex()}" 
            }
        }
        
        response = requests.post("https://replicate.com", json=payload, headers=headers).json()
        
        # Wait and download the generated chunk
        # (For production, implement a short polling loop until prediction status is 'succeeded')
        audio_url = response["output"] 
        audio_data = requests.get(audio_url).content
        
        speech_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Construct the silent backing timeline using your original robust overlay engine
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
