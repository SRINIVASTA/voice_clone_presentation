# --- CRUCIAL PYTHON 3.14 PYDUB IMPORT PATCH ---
import sys
try:
    import audioop
    sys.modules['pyaudioop'] = audioop
except ImportError:
    pass
# -----------------------------------------------

import streamlit as st
import os
from utils import process_presentation

st.set_page_config(page_title="AI Presentation Sync", page_icon="🎤", layout="centered")

st.title("🎤 Perfect-Timed AI Presentation Generator")
st.write("Upload your assets to generate a perfectly synced audio timeline.")

# Initialize generation version tracking to fix browser cache overlapping bugs
if "generation_id" not in st.session_state:
    st.session_state.generation_id = 0

# 1. File Upload Fields
uploaded_txt = st.file_uploader("Upload Presentation Timestamps (.txt)", type=["txt"])
uploaded_voice = st.file_uploader("Upload Your Reference Voice Clone (.wav)", type=["wav"])
ref_text = st.text_input("Reference Text", value="This is a recording of my own voice to train the AI model in Kolab")

# 2. Variable Settings Section
with st.sidebar:
    st.header("⚙️ App Settings")
    # min_value set to 0.0 to make padding completely optional
    padding_time = st.number_input(
        "End padding (seconds)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.0, 
        step=0.5,
        help="Adds a silent buffer at the end of your presentation to prevent abrupt cutoffs. Set to 0.0 for no padding."
    )
    
    # 🔒 Secure User Input field for public web cloud spaces
    user_hf_token = st.text_input(
        "Hugging Face Token (Optional)", 
        type="password", 
        help="Paste a personal token here if public shared channels are busy."
    )

# 3. Compile Sequence Activation
if uploaded_txt and uploaded_voice and ref_text:
    if st.button("🚀 Generate Presentation Audio", type="primary"):
        # Increment version to immediately wipe old, stuck browser audio pipelines
        st.session_state.generation_id += 1
        
        with st.spinner("Processing timeline and stitching audio clips via Cloud API..."):
            try:
                # Save input contents down to localized temporary cloud folders
                with open("temp_presentation.txt", "wb") as f:
                    f.write(uploaded_txt.getbuffer())
                with open("temp_voice.wav", "wb") as f:
                    f.write(uploaded_voice.getbuffer())
                
                output_path = "perfect_timed_presentation.wav"
                
                # Forward files and custom token to our parsing engine
                process_presentation(
                    "temp_presentation.txt", 
                    "temp_voice.wav", 
                    ref_text, 
                    output_path, 
                    padding_time if padding_time is not None else 0.0,
                    token=user_hf_token if user_hf_token else None
                )
                
                # UI success message block once file compilation finishes
                st.success("🎉 Audio successfully compiled via cloud synthesis!")
                
                # Render Audio Output Stream
                # FIXED: Removed 'key' parameter to prevent MediaMixin.audio() error
                with open(output_path, "rb") as audio_file:
                    st.audio(
                        audio_file.read(), 
                        format="audio/wav"
                    )
                
                # Actionable Download Component
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="💾 Download Presentation WAV",
                        data=file,
                        file_name="perfect_timed_presentation.wav",
                        mime="audio/wav",
                        key=f"download_btn_v_{st.session_state.generation_id}"
                    )
            
            # Catch clear formatting/API limit warnings from utils.py directly
            except ValueError as ve:
                st.warning(f"⚠️ API Limit or Format Warning: {ve}")
            except Exception as e:
                st.error(f"❌ An error occurred during generation: {e}")
            finally:
                # Local environmental cleanup loop to prevent filling cloud cache disks
                for tmp_file in ["temp_presentation.txt", "temp_voice.wav"]:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
else:
    st.info("💡 Please upload both your presentation text file and reference voice track to begin.")
