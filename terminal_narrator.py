import torch
import voice_blend
from IPython.display import display, Audio
import soundfile as sf
import audio_joiner as aj
import os
from os.path import join, isfile


# load voice tensors
af_alloy = torch.load('assets/voices/af_sarah.pt')
af_bella = torch.load('assets/voices/am_adam.pt')
af_heart = torch.load('assets/voices/af_heart.pt')

voice_list = [af_alloy, af_bella, af_heart]
weight_list = [0.5, 0.3, 0.2]  # Example weights for blending

# input text
test_txt = """This is a test blending  voices from kokoro.
This is a test blending  voices from kokoro.
This is a test blending  voices from kokoro.
This is a test blending  voices from kokoro."""

new_voice = voice_blend.blending(voice_list, weight_list, test_txt)

# add all files in temp directory to a list
os.makedirs("temp\\", exist_ok=True)

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