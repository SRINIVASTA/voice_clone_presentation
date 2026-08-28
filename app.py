import streamlit as st
import os
from utils import process_presentation

st.set_page_config(page_title="AI Presentation Sync", page_icon="🎤", layout="centered")

st.title("🎤 Perfect-Timed AI Presentation Generator")
st.write("Upload your assets to generate a perfectly synced audio timeline.")

# 1. File Upload Fields
uploaded_txt = st.file_uploader("Upload Presentation Timestamps (.txt)", type=["txt"])
uploaded_voice = st.file_uploader("Upload Your Reference Voice Clone (.wav)", type=["wav"])
ref_text = st.text_input("Reference Text", value="Type the first sentence of your voice recording here.")

# 2. Variable Settings Section
with st.sidebar:
    st.header("⚙️ App Settings")
    padding_time = st.number_input("End padding (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    # 🔒 Secure User Input field (Acts as a text slot on your local desktop app screen)
    user_hf_token = st.text_input(
        "Hugging Face Token (Optional)", 
        type="password", 
        help="Paste a personal token here if public shared channels are busy."
    )

# 3. Compile Sequence Activation
if uploaded_txt and uploaded_voice and ref_text:
    if st.button("🚀 Generate Presentation Audio", type="primary"):
        with st.spinner("Processing timeline and stitching audio clips..."):
            try:
                # Save input contents down to localized temporary files
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
                    padding_time,
                    token=user_hf_token if user_hf_token else None
                )
                
                st.success("🎉 Audio successfully compiled!")
                
                # Render Audio Output Stream
                with open(output_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")
                
                # Actionable Download Component
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
                # Local environmental cleanup loop
                for tmp_file in ["temp_presentation.txt", "temp_voice.wav"]:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
else:
    st.info("💡 Please upload both your presentation text file and reference voice track to begin.")
