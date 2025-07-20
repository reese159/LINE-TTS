import torch
from kokoro import KPipeline

def blending(voice_list, weight_list, text_to_narrate = ""):
    """
    Function to blend voices from Kokoro TTS and save the blended voice.
    This function loads voice tensors, blends them, and returns the resulting voice generator.
    """

    # Create KPipeline for TTS, currently specifying 'a' for American English
    pipeline = KPipeline(lang_code='a', repo_id=r'hexgrad/Kokoro-82M')

    # TODO: MOVE FOLLOWING NOTE TO README
    # Note: the language code refers to the language of the text itself and can be at odds with the voice tensor
    # e.g a british, japanese, etc. voice (bf_george) can be used to speak american english text (lang-code='a')
    
    # Use CUDA-enabled gpu if available, else default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
         # Load voice tensors if they are provided as strings (paths)
        loaded_voices = []
        for voice in voice_list:
            if isinstance(voice, str):
                try:
                    loaded_voice = torch.load(f'assets\\voices\\{voice.strip()}.pt')
                    loaded_voices.append(loaded_voice)
                except FileNotFoundError:
                    print(f"""Voice file {voice.strip()}.pt not found in assets/voices directory.
                          Checking user_voices directory for voice file.""")
                    loaded_voice = torch.load(f'user_voices\\{voice.strip()}.pt')
                    loaded_voices.append(loaded_voice)
                    # return None
            else:
                loaded_voices.append(voice)
        # Convert weights to floats if they are strings
        weight_floats = [float(w) for w in weight_list]
        # Ensure all voices are on the same device
        voices = [voice.to(device) for voice in loaded_voices]

        # Blend voices using the provided weights
        # zip multiplies each voice tensor by its corresponding weight,then we take the summ
        blended_voice = sum(weight * voice for weight, voice in zip(weight_floats, voices))

        # save new voice to pipeline & return after removing extra dimension
        pipeline.voices['blended_voice'] = blended_voice.squeeze(0)

        # return voice generator using new voice, as well as new voice
        return pipeline(
            text=text_to_narrate,
            voice='blended_voice', speed=1,
            # Currently separating text by paragraphs, but can be adjusted
            # Other options: punctuatino or new liner'(?<=[.!?])\s+'#r'\n+'
            split_pattern=r'\n\n'
        ), blended_voice


    except Exception as e:
        print(f"Error blending voices: {e}")
        return None

def blending_pt_files(voice_tensor_list, weight_list, text_to_narrate = ""):
    """
    Function to blend voices from Kokoro TTS and save the blended voice.
    This function loads voice tensors, blends them, and returns the resulting voice generator.
    """

    # Create KPipeline for TTS, currently specifying 'a' for American English
    pipeline = KPipeline(lang_code='a', repo_id=r'hexgrad/Kokoro-82M')

    # TODO: MOVE FOLLOWING NOTE TO README
    # Note: the language code refers to the language of the text itself and can be at odds with the voice tensor
    # e.g a british, japanese, etc. voice (bf_george) can be used to speak american english text (lang-code='a')
    
    # Use CUDA-enabled gpu if available, else default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load voice tensors 
        loaded_voices = []
        for voice in voice_tensor_list:
            if isinstance(voice, str):
                try:
                    # loaded_voice = torch.load(voice)
                    loaded_voices.append(loaded_voice)
                except FileNotFoundError:
                    print(f"""Voice file {voice.strip()}.pt not found in assets/voices directory.
                        Checking user_voices directory for voice file.""")
                    loaded_voice = torch.load(f'user_voices\\{voice.strip()}.pt')
                    loaded_voices.append(loaded_voice)
                    # return None
            else:
                loaded_voices.append(voice)
        # Convert weights to floats if they are strings
        weight_floats = [float(w) for w in weight_list]
        # Ensure all voices are on the same device
        # voices = [voice.to(device) for voice in loaded_voices]

        # Blend voices using the provided weights
        # zip multiplies each voice tensor by its corresponding weight,then we take the summ
        blended_voice = sum(weight * voice for weight, voice in zip(weight_floats, voice_tensor_list))

        # save new voice to pipeline & return after removing extra dimension
        pipeline.voices['blended_voice'] = blended_voice.squeeze(0)

        # return voice generator using new voice, as well as new voice
        return pipeline(
            text=text_to_narrate,
            voice='blended_voice', speed=1,
            # Currently separating text by paragraphs, but can be adjusted
            # Other options: punctuatino or new liner'(?<=[.!?])\s+'#r'\n+'
            split_pattern=r'\n\n'
        ), blended_voice


    except Exception as e:
        print(f"Error blending voices: {e}")
        return None
    
    