import streamlit as st
import torch

st.title("LINE-TTS Narration App")
st.write("Local Interactive Narration Environment")

st.set_page_config(layout="wide")

# Initialize the validation checks
if 'valid_voice' not in st.session_state:
    st.session_state.valid_voice = False
if 'valid_text' not in st.session_state:        # TODO: Validate current text input prior to summary & narration
    st.session_state.valid_text = False

# Initialize voices in session state
if "voices" not in st.session_state:
    st.session_state.voices = []
    
with st.sidebar:
    st.subheader("Config")
    st.write("Adjust the settings below:")
    # Option to change OpenAI model
    model = st.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    # Option to set max tokens for summary
    max_tokens = st.slider("Max Tokens for Summary", min_value=50, max_value=1000, value=250, step=50)
    st.write("Credit to hexgrad for Kokoro-82M voice models and Kokoro inference library")
    st.link_button("Models on HuggingFace", "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices")

# File uploader for the voice tensor file
uploaded_voices = st.file_uploader("Upload voice tensor file (.pt)", type=["pt"], accept_multiple_files=True)

if uploaded_voices:
    # Check for duplicate voice names
    loaded_voice_names = {voice["name"] for voice in st.session_state.voices}
    
    for uploaded_voice in uploaded_voices:
        if uploaded_voice.name not in loaded_voice_names:
            try:
                new_voice = torch.load(uploaded_voice)
                st.session_state.voices.append({"name": uploaded_voice.name,
                                                "data": uploaded_voice, "weight": 0.0})

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
if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        # TODO: read through pdf
elif input_type == "Enter Text":
    text_input = st.text_area("Enter your text here:", height=150)
    if text_input:
        st.success("Text entered successfully!")

# --- Generate summary ---
from text_summarization import summarize_text
st.subheader("Text Summarization")

if st.button("Summarize Text"):
    summary = summarize_text(text_input, model=model, max_tokens=250, openai_api_key=st.secrets["OPENAI_API_KEY"])
    st.write("Summary:")
    st.write(summary)
    
    
