import torch
import voice_blend
from IPython.display import display, Audio
import soundfile as sf
import audio_joiner as aj
import os
# from os.path import join, isfile
import file_reader

# NOTE: This will be made more robust with checks for validity in final version
voices_to_blend = input("Enter the voices to blend (comma-separated, e.g., af_sarah, am_adam, af_heart): ")
voices_to_blend = voices_to_blend.replace(" ", "")  # Remove any spaces
voice_list = voices_to_blend.split(",")  # Split input by commas

print(voice_list)

# NOTE: This will be made more robust with checks for validity in final version
weights_to_blend = input("Enter the weights to blend (comma-separated, e.g., 0.5, 0.3, 0.2): ")
weights_to_blend = weights_to_blend.replace(" ", "")  # Remove any spaces
weight_list = weights_to_blend.split(",")  # Split input by commas

# NOTE: This will be made more robust with checks for validity in final version
pdf_to_narrate = input("Enter the path to the PDF to narrate: ")
text_to_narrate = file_reader.read_pdf(pdf_to_narrate)  # Read the PDF content

print(text_to_narrate)

# load voice tensors from list
# for i, voice_name in enumerate(voice_to_blend):
#     voice_to_blend[i] = torch.load(f'assets/voices/{voice_name.strip()}.pt')

new_voice = voice_blend.blending(voice_list, weight_list, text_to_narrate)

# add all files in temp directory to a list
os.makedirs("temp\\", exist_ok=True)

print(new_voice)

# display and save audio segments using method displayed in kokoro documentation:
for i, (gs, ps, audio) in enumerate(new_voice):
    print(f"Blended Voice - Segment {i}:")
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    display(Audio(data=audio, rate=24000))
    sf.write(f'temp\\blended_voice_segment_{i}.wav', audio, 24000) # save each audio file

# print(f"Temporary audio files: {temp_audio}")  # print list of temp audio files
aj.join_audio_files("temp", narration_name="user_narration_test.wav") # join audio files
aj.clear_temp_files() # clear temp files after joining