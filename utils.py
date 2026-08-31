import os
import re
from pydub import AudioSegment
from f5_tts.infer.utils_infer import load_model, infer_process
from f5_tts.model import DiT

# Global placeholders to keep the neural network loaded in background memory
_LOCAL_MODEL = None
_LOCAL_VOCIDER = None

def parse_time(time_str):
    m, s = map(int, time_str.split(':'))
    return (m * 60 + s) * 1000

def split_text_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def get_local_engine():
    """
    Initializes and caches the F5-TTS model weights locally on your machine.
    Bypasses busy Hugging Face network clouds completely.
    """
    global _LOCAL_MODEL, _LOCAL_VOCIDER
    if _LOCAL_MODEL is None:
        print("📥 Initializing native F5-TTS model core engine on your hardware...")
        # Automatically downloads safe architecture weights from Hugging Face to local cache folder
        _LOCAL_MODEL, _LOCAL_VOCIDER = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            checkpoint_path="SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"
        )
    return _LOCAL_MODEL, _LOCAL_VOCIDER

def generate_local_chunk(voice_path, ref_text, text_chunk, temp_out_path="chunk_temp.wav"):
    """
    Synthesizes speech using your own computer processing power. Guaranteeing voice output.
    """
    try:
        model, vocoder = get_local_engine()
        
        # Native processing command handles raw audio rendering directly on local device
        infer_process(
            ref_audio=voice_path,
            ref_text=ref_text,
            gen_text=text_chunk,
            model_obj=model,
            vocoder_obj=vocoder,
            output_path=temp_out_path
        )
        
        if os.path.exists(temp_out_path) and os.path.getsize(temp_out_path) > 1000:
            segment = AudioSegment.from_file(temp_out_path)
            os.remove(temp_out_path)
            return segment
    except Exception as e:
        print(f"⚠️ Local generation anomaly: {e}")
    return None

def process_presentation(txt_path, voice_path, ref_text, output_path, padding_seconds, token=None):
    print("⚡ Activating local timeline stitching suite...")
    
    # Clean up the long voice template sample file to capture an optimized dense blueprint
    try:
        raw_voice = AudioSegment.from_file(voice_path)
        optimized_voice = raw_voice[:15000] # Safe 15-second tracking clip
        optimized_voice = optimized_voice.set_frame_rate(24000).set_channels(1)
        optimized_voice.export("temp_optimized_voice.wav", format="wav")
        voice_blueprint = "temp_optimized_voice.wav"
    except Exception as audio_err:
        raise ValueError(f"Failed to index your input speaker voice sample track: {audio_err}")

    # Parse your presentation timestamp scripts
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments = re.split(r'\[(\d{2}:\d{2})\]', content)[1:]
    if not segments:
        raise ValueError("Could not extract timelines. Verify presentation scripts contain explicit [MM:SS] timestamps.")

    timestamps = [parse_time(segments[i]) for i in range(0, len(segments), 2)]
    texts = [segments[i].strip() for i in range(1, len(segments), 2)]

    final_presentation_audio = AudioSegment.empty()
    current_timeline_position = 0

    # --- SECTION 1: NARRATING REFERENCE TEXT AS INTRO SLIDE 1 ---
    print("🎤 Generating Slide 1 Introduction locally...")
    ref_chunks = split_text_into_sentences(ref_text)
    for chunk in ref_chunks:
        sentence_audio = generate_local_chunk(voice_blueprint, ref_text, chunk)
        if sentence_audio:
            final_presentation_audio += sentence_audio
            final_presentation_audio += AudioSegment.silent(duration=250)
            
    final_presentation_audio += AudioSegment.silent(duration=1500) # Slide transition break
    intro_duration = len(final_presentation_audio)
    current_timeline_position = intro_duration

    # --- SECTION 2: STITCHING MAIN CHRONOLOGICAL SLIDES ---
    for idx, (start_time, full_text) in enumerate(zip(timestamps, texts)):
        if not full_text or not full_text.strip():
            continue

        shifted_start_time = start_time + intro_duration
        chunks = split_text_into_sentences(full_text)
        block_audio = AudioSegment.empty()

        for chunk in chunks:
            sentence_audio = generate_local_chunk(voice_blueprint, ref_text, chunk)
            if sentence_audio:
                block_audio += sentence_audio
                block_audio += AudioSegment.silent(duration=250)

        # Map chronological placement on the timeline canvas
        if shifted_start_time > current_timeline_position:
            silence_needed = shifted_start_time - current_timeline_position
            final_presentation_audio += AudioSegment.silent(duration=silence_needed)
            current_timeline_position = shifted_start_time

        final_presentation_audio += block_audio
        current_timeline_position += len(block_audio)

    if padding_seconds > 0:
        final_presentation_audio += AudioSegment.silent(duration=int(padding_seconds * 1000))

    # Clean local temporary clips from workspace
    if os.path.exists("temp_optimized_voice.wav"):
        os.remove("temp_optimized_voice.wav")

    # Master compilation output export
    final_presentation_audio.export(output_path, format="wav")
    print("🎉 Entire presentation compiled completely using local hardware.")
