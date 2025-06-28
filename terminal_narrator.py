import torch
import voice_blend

# load voice tensors
af_alloy = torch.load('assets/voices/af_sarah.pt')
af_bella = torch.load('assets/voices/am_adam.pt')
af_heart = torch.load('assets/voices/af_heart.pt')

voice_list = [af_alloy, af_bella, af_heart]
weight_list = [0.5, 0.3, 0.2]  # Example weights for blending

new_voice = voice_blend(voice_list, weight_list)

# input text
test_txt = """This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro.
This is a test blending the voices of alloy and bella from kokoro."""


