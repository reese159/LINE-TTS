import streamlit as st
import torch

st.title("LINE-TTS Narration App")
st.write("Local Interactive Narration Environment")

if "voices" not in st.session_state:
    st.session_state.voices = []


# File uploader for the voice tensor file
uploaded_voice = st.file_uploader("Upload voice tensor file (.pt)", type=["pt"], accept_multiple_files=True)

if uploaded_voice:
    try:
        new_voice = torch.load(uploaded_voice)
        if st.session_state.voices.get(uploaded_voice.name,) is not None:
            st.session_state.voices.append({"name": uploaded_voice.name,
                                            "data": uploaded_voice, "weight": 0.0})
            st.success("Voice tensor loaded successfully!")
        else:
            st.error("The uploaded file does not contain a valid voice tensor.")

    except Exception as e:
        st.error(f"Error loading voice tensor: {e}")

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