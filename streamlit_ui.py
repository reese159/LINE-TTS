import streamlit as st
import torch

st.title("LINE-TTS Narration App")
st.write("Local Interactive Narration Environment")

if "voices" not in st.session_state:
    st.session_state.voices = []


# File uploader for the voice tensor file
uploaded_voice = st.file_uploader("Upload voice tensor file (.pt)", type=["pt"])

st.session_state.voices = []
st.session_state.weights = []
if uploaded_voice is not None:
    try:
        new_voice = torch.load(uploaded_voice)
        st.session_state.voices.append({"name": uploaded_voice.name, "data": uploaded_voice})
        st.success("Voice tensor loaded successfully!")
    except Exception as e:
        st.error(f"Error loading voice tensor: {e}")

for voice in st.session_state.voices:
    weight = st.number_input("Select a weight for the voice tensor (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    st.session_state.weights.append(weight)

if sum(st.session_state.weights) != 1.0:
    st.warning("The sum of the weights must equal 1.0. Please adjust the weights accordingly.")
