import os
import re
from pydub import AudioSegment
from gradio_client import Client, handle_file

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    # Establish handshake with the open cluster cluster nodes
    print("🛰️ Connecting to F5-TTS compute pipeline...")
    
    if token:
        # Connect safely using the user-pasted credential parameter key
        client = Client("mrfakename/E2-F5-TTS", token=token)
    else:
        # Fallback to shared open infrastructure tunnels
        client = Client("mrfakename/E2-F5-TTS")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    generated_segments = []
    max_required_duration = 0

    wrapped_voice = handle_file(voice_path)

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        print(f"🎤 Synthesizing node segment for timeline position {start_time/1000}s...")
        
        # Pull response arrays out of active prediction frames
        result = client.predict(
            ref_audio=wrapped_voice,
            ref_text=ref_text,
            gen_text=speech_text,
            remove_silence=False,
            api_name="/predict"
        )
        
        remote_wav_path = result if isinstance(result, (list, tuple)) else result
        
        # Load local sound properties into timeline matrices
        speech_segment = AudioSegment.from_file(remote_wav_path, format="wav")
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Construct the silent backing layer timeline
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    # Position each sound module systematically to prevent layout drift
    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    print("🎉 Master audio output created successfully.")
