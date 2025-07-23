import streamlit as st
import torch
import soundfile as sf
import audio_joiner as aj
import voice_blend
import file_reader
import io
from IPython.display import display, Audio
from pydub import AudioSegment
from text_summarization import summarize_text

st.set_page_config(layout="wide")

st.title("LINE-TTS Narration App")
st.write("Local Interactive Narration Environment")

# Initialize the validation checks
if 'valid_voice' not in st.session_state:
    st.session_state.valid_voice = False
if 'text_input' not in st.session_state:        # TODO: Validate current text input prior to summary & narration
    st.session_state.text_input = ""


# Initialize voices in session state
if "voices" not in st.session_state:
    st.session_state.voices = []
# if "current_weights" not in st.session_state:
#     st.session_state.current_weights = []

with st.sidebar:
    st.subheader("Config")
    st.write("Adjust the settings below:")
    # Option to change OpenAI model
    model = st.selectbox("Select OpenAI Model", ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1"], index=0)
    # Option to set max tokens for summary
    max_tokens = st.slider("Max Tokens for Summary Output", min_value=50, max_value=750, value=250, step=50)
    st.write("Credit to hexgrad for Kokoro-82M voice models and Kokoro inference library")
    st.link_button("Models on HuggingFace", "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices")


# Use st.cache_resource to cache loaded voice tensors across reruns
@st.cache_resource
def load_voice_tensor(uploaded_file_object):
    """Loads a voice tensor from an uploaded file and moves it to the appropriate device."""
    try:
        # Streamlit's file_uploader provides a SpooledTemporaryFile, which torch.load can handle.
        loaded_tensor = torch.load(uploaded_file_object)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return loaded_tensor.to(device)
    except Exception as e:
        st.error(f"Error loading voice tensor: {e}")
        return None

# File uploader for the voice tensor file
uploaded_voices = st.file_uploader("Upload voice tensor file (.pt)", type=["pt"], accept_multiple_files=True)

if uploaded_voices:
    # Check for duplicate voice names
    loaded_voice_names = {voice["name"] for voice in st.session_state.voices}
    
    for uploaded_voice in uploaded_voices:
        if uploaded_voice.name not in loaded_voice_names:
            try:
                # loaded_voice = torch.load(uploaded_voice)
                loaded_voice = load_voice_tensor(uploaded_voice)
                # new_voice = torch.load(uploaded_voice)
                st.session_state.voices.append({"name": uploaded_voice.name,
                                                "tensor": loaded_voice, "weight": 0.0})

            except Exception as e:
                st.error(f"Error loading voice tensor {uploaded_voice.name}: {e}")
                
    st.success("Voice tensors loaded!")

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

# --- Text input for narration ---
st.subheader("Text Input for Narration")
input_type = st.radio("Choose Input Type:", ("Upload PDF", "Enter Text"))

# Cached function for reading PDF

@st.cache_data
def read_pdf_content(pdf_file_buffer):
    # Reads content from a PDF file buffer
    return file_reader.read_pdf(pdf_file_buffer)

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        pdf_document = io.BytesIO(bytes_data)
        st.session_state.text_input = read_pdf_content(pdf_document)  # Read the PDF content
        st.success("PDF file uploaded successfully!")
elif input_type == "Enter Text":
    st.session_state.text_input = st.text_area("Enter your text here:", height=150)
    if st.session_state.text_input:
        st.success("Text entered successfully!")

# --- Generate summary ---
summarization_area, narration_area = st.columns(2)

@st.cache_resource(show_spinner=False)
def cached_blending(_voices, weights, text):
    return voice_blend.blending_pt_files(_voices, weights, text)


if st.session_state.text_input:
    with summarization_area:
        st.subheader("Text Summarization")

        if st.button("Summarize Text"):
            
            log_narration=""
            
            with st.spinner("Generating summary..."):
                summary = summarize_text(st.session_state.text_input, model=model, openai_api_key=st.secrets["OPENAI_API_KEY"])
                st.text_area("Sumary:", summary, height=150)
                narration_text_box = st.empty()
                
                
                current_voices = [voice["tensor"] for voice in st.session_state.voices]
                current_weights = [voice["weight"] for voice in st.session_state.voices]
                new_pipeline, new_voice = cached_blending(current_voices, current_weights, summary)
                
                summary_audio = AudioSegment.empty()
                
                # narration_text_box.text_area("Watch the narration process:", "", height=150)
            
            with st.spinner("Generating summary narration..."):
                # display and save audio segments using method displayed in kokoro documentation:
                for i, (gs, ps, audio) in enumerate(new_pipeline):
                    log_narration = f"""Segment {i}:
Graphemes: {gs}
Phonemes: {ps}
""" + log_narration
                    narration_text_box.text_area("Watch the narration process:", log_narration, height=150)
                    new_audio_segment  = aj.tensor_to_audio_segment(audio, sample_rate=24000)
                    summary_audio += new_audio_segment
                    print(log_narration)
                    # st.audio #add_audio_to_narration(temp_path=NotImplemented, narration_name="user_narration.wav", new_audio=audio)
                audio_buffer = io.BytesIO()
                summary_audio.export(audio_buffer, format="wav") 
                st.audio(data=audio_buffer)
            
    with narration_area:
        st.subheader("Full Narration")
        
        if st.button("Generate Full Narration"):
            log_narration = ""
            
            with st.spinner("Generating full narration..."):
                narration_text_box = st.empty()
                current_voices = [voice["tensor"] for voice in st.session_state.voices]
                current_weights = [voice["weight"] for voice in st.session_state.voices]
                new_pipeline, new_voice = cached_blending(current_voices, current_weights, st.session_state.text_input)
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
                        print(log_narration)
                
            audio_buffer = io.BytesIO()
            full_audio.export(audio_buffer, format="wav") 
            st.audio(data=audio_buffer)
