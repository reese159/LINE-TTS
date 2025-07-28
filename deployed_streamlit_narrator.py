import streamlit as st
import torch
import soundfile as sf
import audio_joiner as aj
import voice_blend
import file_reader
import io
import os
from IPython.display import display, Audio
from pydub import AudioSegment

# --- Streamlit configuration ---
# Initialize the validation checks
if 'valid_voice' not in st.session_state:
    st.session_state.valid_voice = False
if 'text_input' not in st.session_state:        # TODO: Validate current text input prior to summary & narration
    st.session_state.text_input = ""
# Initialize voices in session state
if "voices" not in st.session_state:
    st.session_state.voices = []
# Set statics and page config
MAX_CHARS_LIMIT = 500
user_instructions = f"""This application allows you to blend voices from Kokoro TTS, creating your own custom voices and generating narrations.
The application is meant to help meet accessibility needs, allowing users the option to generate audio from text-based content locally without relying on subscription-based services.
This helps to ensure that users can access content in a way that is convenient, cost-effective, and privacy-minded.
This community cloud version only allows for a maximum of {MAX_CHARS_LIMIT} characters when generating narrations to avoid exceeding resource limits. 
For larger text/documents, please download the freely available repository and use the local version."""
st.set_page_config(layout="wide", page_title="LINE-TTS Narration App", page_icon=":microphone:")
st.title("LINE-TTS Narration App")

# --- Sidebar for configuration ---
with st.sidebar:
    # Option to set max tokens for summary
    st.header("Local Interactive Narration Environment")
    st.divider()
    st.subheader("AI Summary Config")
    st.write("Credit to OpenAI for the GPT models used in this app.")
    # Option to change OpenAI model
    model = st.selectbox("Select OpenAI Model", ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1"], index=0)
    max_tokens = st.slider("Max Tokens for Summary Output Provided by GPT", min_value=50, max_value=500, value=250, step=50)
    st.divider()
    st.write("Credit to hexgrad for Kokoro-82M voice models and Kokoro inference library. Please follow the link below to freely download and access the voice tensors on HuggingFace.")
    st.link_button("Voices", "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices")
st.subheader("Voice Selection and Blending")

# --- Select from existing voices ---
try:
    voice_dir = "assets/voices"
    if os.path.exists(voice_dir) and os.path.isdir(voice_dir):
        existing_voices_options = [f for f in os.listdir(voice_dir) if f.endswith('.pt')]
    else:
        existing_voices_options = []
        st.info("`assets/voices` directory not found. Place pre-existing voices there to select them.")
except Exception as e:
    existing_voices_options = []
    st.warning(f"Could not read `assets/voices` directory: {e}")

# --- voice selection ---
# refresh voices in session state
st.session_state.voices = []
# function to update multiselect voices
def update_multiselect_voices(selected_existing_voices):
    loaded_voice_names = {voice["name"] for voice in st.session_state.voices}
    for voice_name in selected_existing_voices:
        if voice_name not in loaded_voice_names:
            try:
                voice_path = os.path.join(voice_dir, voice_name)
                loaded_voice = torch.load(voice_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                st.session_state.voices.append({"name": voice_name, "tensor": loaded_voice, "weight": 0.0})
            except Exception as e:
                st.error(f"Error loading existing voice tensor {voice_name}: {e}")

# function to update user uploaded voices
def update_uploaded_voices(uploaded_files):
    for uploaded_file in uploaded_files:
        try:
            loaded_voice = torch.load(uploaded_file).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            voice_name = uploaded_file.name
            # if voice_name not in [voice["name"] for voice in st.session_state.voices]:
            st.session_state.voices.append({"name": voice_name, "tensor": loaded_voice, "weight": 0.0})
        except Exception as e:
            st.error(f"Error loading uploaded voice tensor {uploaded_file.name}: {e}")

# Allow user to select from existing voices
if existing_voices_options:
    selected_existing_voices = st.multiselect(
        "Select from Kokoro-82M voices",
        options=existing_voices_options,
        help="Select from voices available in the `assets/voices` directory."
    )
    # if selected_existing_voices:
    update_multiselect_voices(selected_existing_voices)

uploaded_voices = st.file_uploader("User-uploaded voice tensor file (.pt)", type=["pt"], accept_multiple_files=True)
# Allow user to upload voices and add them to the session state
if uploaded_voices:
    # Check for duplicate voice names
    loaded_voice_names = {voice["name"] for voice in st.session_state.voices}
    update_uploaded_voices(uploaded_voices)

# --- Display loaded voices and get weights ---
if st.session_state.voices:
    st.subheader("Add Voice Weights")
    
    current_weights = []
    # iterate through all voices currently in state
    for i, voice in enumerate(st.session_state.voices):
        weight = st.number_input(
            f"Weight for '{voice['name']}'",
            min_value=0.0,
            max_value=1.0,
            value=voice["weight"], # Use the current weight from session state
            step=0.01,
            key=f"weight_input_{voice['name']}_{i}" # Unique key
        )
    # Update the weight in the session state
        st.session_state.voices[i]["weight"] = weight
        current_weights.append(weight)
        
    # Validate sum of weights to be 1.0
    total_weight = sum(current_weights)
    if total_weight != 1.0:
        st.session_state.valid_voice = False
        st.warning(f"Total weight is {total_weight:.2f}. It should be 1.0 for proper blending.")
    else:
        st.session_state.valid_voice = True
        st.success("Weights are valid and sum to 1.0.")
st.divider()

# --- Text input for narration ---
st.subheader("Text Input for Narration")
st.markdown(f"*User Note* - The application currently supports a maximum of {MAX_CHARS_LIMIT} characters for full narration")
input_type = st.radio("Choose Input Type:", ("Enter Text", "Upload PDF"))
if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        pdf_document = io.BytesIO(bytes_data)
        st.session_state.text_input = file_reader.read_pdf(pdf_document)  # Read the PDF content
        st.success("PDF file uploaded successfully!")
elif input_type == "Enter Text":
    st.session_state.text_input = st.text_area("Enter your text here:", "This text will be narrated by the user-selected voice", max_chars=MAX_CHARS_LIMIT,
    help=f"Maximum {MAX_CHARS_LIMIT} characters allowed.", height=150)
    if st.session_state.text_input:
        st.success("Text entered successfully!")

# --- Generate summary ---
from text_summarization import summarize_text
st.divider()
summarization_area, narration_area = st.columns(2)

# --- narration area ---
if st.session_state.text_input and st.session_state.valid_voice:
    #--- Summarization area ---
    with summarization_area:
        st.subheader("Text Summarization")
        if st.button("Summarize Text"):
            log_narration=""
            
            # Start genertaing summarization
            with st.spinner("Generating summary..."):
                summary = summarize_text(st.session_state.text_input, model=model, openai_api_key=st.secrets["OPENAI_API_KEY"])
                st.text_area("Sumary:", summary, height=150)
                narration_text_box = st.empty()
                current_voices = [voice["tensor"] for voice in st.session_state.voices]
                current_weights = [voice["weight"] for voice in st.session_state.voices]
                new_pipeline, new_voice = voice_blend.blending_pt_files(current_voices, current_weights, summary)
                summary_audio = AudioSegment.empty()
            # Start generating summary narration
            with st.spinner("Generating summary narration..."):
                # display and save audio segments using method displayed in kokoro documentation:
                for i, (gs, ps, audio) in enumerate(new_pipeline):
                    log_narration = f"""Segment {i}:
Graphemes: {gs}
Phonemes: {ps}
""" + log_narration
                    narration_text_box.text_area("Watch the narration process:", log_narration, height=500)
                    new_audio_segment  = aj.tensor_to_audio_segment(audio, sample_rate=24000)
                    summary_audio += new_audio_segment
                # Display audio generated
                audio_buffer = io.BytesIO()
                summary_audio.export(audio_buffer, format="wav") 
                st.audio(data=audio_buffer)
                # Allow user to save the blended voice tensor to a file
                voice_buffer = io.BytesIO()
                torch.save(new_voice, voice_buffer)
                blended_voice_file_name = st.text_input("Enter desired voice file name (e.g.,my_voice.pt):", "new_voice.pt")
                st.download_button(label="Save Current Voices and Weights",
                                data=voice_buffer,
                                file_name=blended_voice_file_name)
    
    # --- Full narration area ---
    with narration_area:
        st.subheader("Full Narration")
        if st.button("Generate Full Narration"):
            log_narration = ""
            # Generate full narration
            with st.spinner("Generating full narration..."):
                narration_text_box = st.empty()
                # Update voices, prep for narration
                current_voices = [voice["tensor"] for voice in st.session_state.voices]
                current_weights = [voice["weight"] for voice in st.session_state.voices]
                truncated_text = st.session_state.text_input[:MAX_CHARS_LIMIT]  # Truncate to first MAX_CHARS_LIMIT characters for processing
                new_pipeline, new_voice = voice_blend.blending_pt_files(current_voices, current_weights, truncated_text)
                full_audio = AudioSegment.empty()
                
                # display and save audio segments using method displayed in kokoro documentation:
                for i, (gs, ps, audio) in enumerate(new_pipeline):
                        log_narration = f"""Segment {i}:
    Graphemes: {gs}
    Phonemes: {ps}
    """ + log_narration
                        narration_text_box.text_area("Watch the narration process:", log_narration, height=500)
                        new_audio_segment = aj.tensor_to_audio_segment(audio, sample_rate=24000)
                        full_audio += new_audio_segment
            # Display audio generated
            audio_buffer = io.BytesIO()
            full_audio.export(audio_buffer, format="wav") 
            st.audio(data=audio_buffer)
            # Allow user to save the blended voice tensor to a file
            voice_buffer = io.BytesIO()
            torch.save(new_voice, voice_buffer)
            blended_voice_file_name = st.text_input("Enter desired voice file name (e.g.,my_voice.pt):", "new_voice.pt")
            st.download_button(label="Save Current Voices and Weights",
                            data=voice_buffer,
                            file_name=blended_voice_file_name)

# --- Page Footer ---
st.divider() 
st.markdown(user_instructions)
st.link_button("Local Version Download", "https://github.com/reese159/LINE-TTS")