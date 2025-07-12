import streamlit as st
import torch

st.title("LINE-TTS Narration App")
st.write("Local Interactive Narration Environment")

# Initialize the validation checks
if 'valid_voice' not in st.session_state:
    st.session_state.valid_voice = False
if 'valid_text' not in st.session_state:
    st.session_state.valid_text = False

# Initialize voices in session state
if "voices" not in st.session_state:
    st.session_state.voices = []

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
        st.warning(f"Total weight is {total_weight:.2f}. It should be 1.0 for proper blending.")
    else:
        st.success("Weights are valid and sum to 1.0.")

# --- Text input for narration ---
st.subheader("Text Input for Narration")
input_type = st.radio("Choose Input Type:", ("Upload PDF", "Enter Text"))
if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        # Process the uploaded PDF file
        st.success("PDF file uploaded successfully!")
        # Add code here to process the uploaded file
elif input_type == "Enter Text":
    text_input = st.text_area("Enter your text here:", height=150)
    if text_input:
        # Process the entered text
        st.success("Text entered successfully!")
        # Add code here to process the text

# --- Generate summary ---