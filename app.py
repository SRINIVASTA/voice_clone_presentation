import streamlit as st
import os
import re
from utils import process_presentation

st.set_page_config(page_title="AI Presentation Sync", page_icon="🎤", layout="centered")

st.title("🎤 Perfect-Timed AI Presentation Generator")
st.write("Upload your assets to generate a perfectly synced audio timeline.")

# 1. File Uploaders
uploaded_txt = st.file_uploader("Upload Presentation Timestamps (.txt)", type=["txt"])
uploaded_voice = st.file_uploader("Upload Your Reference Voice Clone (.wav)", type=["wav"])
ref_text = st.text_input("Reference Text", value="Type the first sentence of your voice recording here.")

# 2. Settings Block
with st.expander("⚙️ Advanced Settings"):
    padding_time = st.number_input("End padding (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# 3. Processing Action
if uploaded_txt and uploaded_voice and ref_text:
    if st.button("🚀 Generate Presentation Audio", type="primary"):
        with st.spinner("Processing timeline and generating AI speech..."):
            try:
                # Save uploaded files temporarily
                with open("temp_presentation.txt", "wb") as f:
                    f.write(uploaded_txt.getbuffer())
                with open("temp_voice.wav", "wb") as f:
                    f.write(uploaded_voice.getbuffer())
                
                # Execute core audio matching logic
                output_path = "perfect_timed_presentation.wav"
                process_presentation("temp_presentation.txt", "temp_voice.wav", ref_text, output_path, padding_time)
                
                st.success("🎉 Audio successfully compiled!")
                
                # Render Audio Player Component
                with open(output_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                
                # Render Download Button Component
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="💾 Download Presentation WAV",
                        data=file,
                        file_name="perfect_timed_presentation.wav",
                        mime="audio/wav"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
            finally:
                # Clean up local temporary files
                for tmp_file in ["temp_presentation.txt", "temp_voice.wav"]:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
else:
    st.info("💡 Please upload both your presentation text file and reference voice track to begin.")
