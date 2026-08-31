import io
import os
import re
from pydub import AudioSegment
from gradio_client import Client, file

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def generate_single_chunk(client, voice_path, ref_text, text_chunk):
    """
    Leverages gradio_client to synthesize zero-shot TTS blocks natively.
    """
    try:
        # Connect explicitly to the official F5-TTS structural processing function
        result = client.predict(
            ref_audio_input=file(voice_path),
            ref_text_input=str(ref_text).strip(),
            gen_text_input=str(text_chunk).strip(),
            remove_silence=False,
            cross_fade_duration=0.15,
            speed=1.0,
            api_name="/infer"
        )
        
        # Gradio spaces return a string path containing the temporary audio file path
        if isinstance(result, (list, tuple)) and len(result) > 0:
            generated_audio_path = result[0]
            if os.path.exists(generated_audio_path):
                return AudioSegment.from_file(generated_audio_path)
                
    except Exception as e:
        print(f"⚠️ Gradio API execution segment timeout error: {e}")
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("🛰️ Connecting via Gradio Cloud Streaming Client...")
    
    clean_token = token.strip() if (token and token.strip()) else None
    
    try:
        # 🌟 FIXED: Gracefully falls back to 'token' to resolve the Client.init() crash
        if clean_token:
            try:
                client = Client("m-a-p/F5-TTS", token=clean_token)
            except TypeError:
                client = Client("m-a-p/F5-TTS", hf_token=clean_token)
        else:
            client = Client("m-a-p/F5-TTS")
    except Exception as e:
        raise ValueError(f"Failed to securely handshake with the Gradio API cluster: {e}")

    # --- AUDIO BLUEPRINT PRE-PROCESSING ---
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        # Trim down reference frame size to 10 seconds to keep pipeline processing fast
        optimized_voice = raw_voice[:10000]
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        
        optimized_voice_path = "optimized_ref_voice.wav"
        optimized_voice.export(optimized_voice_path, format="wav", codec="pcm_s16le")
    except Exception as audio_err:
        raise ValueError(f"Failed to normalize your input reference sound track: {audio_err}")

    # --- READ TIMELINE SCRIPTS ---
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        if os.path.exists(optimized_voice_path):
            os.remove(optimized_voice_path)
        raise ValueError("Timeline structure is invalid. Please format scripts with explicit [MM:SS] timestamps.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: NARRATING THE REFERENCE OVERLAY AS SLIDE 1 ---
    print("🎤 Generating Slide 1 Intro narration via active websocket stream...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_single_chunk(client, optimized_voice_path, ref_text, chunk)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=200)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Slide change pause
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- SECTION 2: COMPILING SUBSEQUENT MAIN SLIDES ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            # Clean up acronyms on the fly so the acoustic vocal cords can pronounce them smoothly
            clean_chunk = chunk.replace("ML", "machine learning").replace("UI", "user interface").replace("JS", "javascript").replace("PDF", "document report")
            
            sentence_audio = generate_single_chunk(client, optimized_voice_path, ref_text, clean_chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=200)

        # Stitch chronological silence gaps on our master audio timeline map
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    # Clean up workspace temporary file assets
    if os.path.exists(optimized_voice_path):
        os.remove(optimized_voice_path)

    # Catch empty server drops
    if len(final_presentation_audio) <= intro_duration + 500:
        raise ValueError("The public Gradio inference worker threads are completely congested. Please wait a brief moment and click generate again.")

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Master Waveform Serialization export
    final_presentation_audio.export(output_path, format="wav", codec="pcm_s16le")
    print("🎉 File timeline fully processed.")
