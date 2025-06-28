import torch
from kokoro import KPipeline

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

def blend_voices(voice_list, weight_list):
    """
    Function to blend voices from Kokoro TTS and save the blended voice.
    This function loads voice tensors, blends them, and returns the resulting voice generator.
    """

    # Create KPipeline for TTS, currently specifying 'a' for American English
    pipeline = KPipeline(lang_code='a', repo_id=r'hexgrad/Kokoro-82M')

    # TODO: MOVE FOLLOWING NOTE TO README
    # Note: the language code refers to the language of the text itself and can be at odds with the voice tensor
    # e.g a british, japanese, etc. voice (bf_george) can be used to speak american english text (lang-code='a')

    # load voice tensors
    af_alloy = torch.load('assets/voices/af_sarah.pt')
    af_bella = torch.load('assets/voices/am_adam.pt')
    af_heart = torch.load('assets/voices/af_heart.pt')

    # Use CUDA-enabled gpu if available, else default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Ensure all voices are on the same device
        voices = [voice.to(device) for voice in voice_list]

        # Blend voices using the provided weights
        # zip multiplies each voice tensor by its corresponding weight,then we take the summ
        blended_voice = sum(weight * voice for weight, voice in zip(weight_list, voices))

        # save new voice to pipeline & return after removing extra dimension
        pipeline.voices['blended_voice'] = blended_voice.squeeze(0)

        # return voice generator using new voice
        return pipeline(
            test_txt, voice='blended_voice',
            speed=1, 
        )#split_pattern=r'\n+'


    except Exception as e:
        print(f"Error blending voices: {e}")
        return None

