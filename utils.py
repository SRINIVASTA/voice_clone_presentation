import os
import re
import io
from pydub import AudioSegment
from gradio_client import Client, handle_file

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds):
    # Connecting directly to the verified open F5 cluster node
    print("🛰️ Opening secure tunnel to F5-TTS cluster...")
    client = Client("mrfakename/E2-F5-TTS")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    generated_segments = []
    max_required_duration = 0

    # Package your voice track using the official Gradio file handler
    wrapped_voice = handle_file(voice_path)

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        print(f"🎤 Remote generating text chunk for slot {start_time/1000}s...")
        
        # Call the accurate API path route directly
        result = client.predict(
            ref_audio=wrapped_voice,
            ref_text=ref_text,
            gen_text=speech_text,
            remove_silence=False,
            api_name="/predict"
        )
        
        # The result returns a list or direct string path depending on chunk layout
        remote_wav_path = result[0] if isinstance(result, (list, tuple)) else result
        
        # Load the generated chunk into our timeline system canvas
        speech_segment = AudioSegment.from_file(remote_wav_path, format="wav")
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Build the silent canvas timeline layer
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    # Map the generated chunks precisely onto the timeline coordinates
    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    print("🎉 Sync sequence complete!")
