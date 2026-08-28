import os
import re
import io
import shutil
from pydub import AudioSegment
from gradio_client import Client, handle_file

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds):
    # Establish a handshake with an alternative, premium open-access F5-TTS cluster
    print("🛰️ Opening secure tunnel to F5-TTS infrastructure...")
    client = Client("mrfakename/F5-TTS")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    generated_segments = []
    max_required_duration = 0

    # Package your voice track using the official Gradio file wrapper
    wrapped_voice = handle_file(voice_path)

    for start_time, speech_text in zip(timestamps, texts):
        if not speech_text: 
            continue

        print(f"🎤 Remote generating text chunk for slot {start_time/1000}s...")
        
        # Trigger generation using the endpoint structure
        result = client.predict(
            ref_audio_input=wrapped_voice,
            ref_text_input=ref_text,
            gen_text_input=speech_text,
            remove_silence=False,
            cross_fade_duration=0.15,
            n_scale_ratio=1.0,
            api_name="/infer"
        )
        
        # The API outputs a tuple where index 0 is the physical path to the complete .wav file
        remote_wav_path = result[0]
        
        # Load the bytes into our overlay system canvas
        speech_segment = AudioSegment.from_file(remote_wav_path, format="wav")
        generated_segments.append((start_time, speech_segment))
        
        end_position = start_time + len(speech_segment)
        if end_position > max_required_duration:
            max_required_duration = end_position

    # Build our robust silent canvas timeline
    padding_ms = int(padding_seconds * 1000)
    final_presentation_audio = AudioSegment.silent(duration=max_required_duration + padding_ms)

    # Map the generated chunks precisely onto the timeline coordinates
    for start_time, speech_segment in generated_segments:
        final_presentation_audio = final_presentation_audio.overlay(speech_segment, position=start_time)

    final_presentation_audio.export(output_path, format="wav")
    print("🎉 Sync sequence complete!")
