import torch
import voice_blend
from IPython.display import display, Audio
import soundfile as sf
import audio_joiner as aj
import os
import file_reader
import text_summarization as ts

# NOTE: This will be made more robust with checks for validity in final version
voices_to_blend = input("Enter the voices to blend (comma-separated, e.g., af_sarah, am_adam, af_heart): ")
voices_to_blend = voices_to_blend.replace(" ", "")  # Remove any spaces
voice_list = voices_to_blend.split(",")  # Split input by commas

# NOTE: This will be made more robust with checks for validity in final version
weights_to_blend = input("Enter the weights to blend (comma-separated, e.g., 0.5, 0.3, 0.2): ")
weights_to_blend = weights_to_blend.replace(" ", "")  # Remove any spaces
weight_list = weights_to_blend.split(",")  # Split input by commas

# NOTE: This will be made more robust with checks for validity in final version
pdf_to_narrate = input("Enter the path to the PDF to narrate: ")
text_to_narrate = file_reader.read_pdf(pdf_to_narrate)  # Read the PDF content

# add all files in temp directory to a list
os.makedirs("temp\\", exist_ok=True)

summarize_text = input("Do you want a summary of the text using GPT? (y/n): ").strip().lower()
if summarize_text == 'y':
    summary_text = ts.summarize_text(text_to_narrate)
    
    new_pipeline, new_voice = voice_blend.blending(voice_list, weight_list, summary_text)
    
    # display and save audio segments using method displayed in kokoro documentation:
    for i, (gs, ps, audio) in enumerate(new_pipeline):
        print(f"Blended Voice - Segment {i}:")
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        display(Audio(data=audio, rate=24000))
    sf.write(f'temp\\blended_voice_segment_{i}.wav', audio, 24000) # save each audio file
    base_name = os.path.splitext(pdf_to_narrate)[0]
    aj.join_audio_files("temp", narration_name=f"{base_name}_summary.wav") # join audio files
    aj.clear_temp_files() # clear temp files after joining
    print(f"summary: {summary_text}")
    print(f"summary narration saved in final_narrations\\{base_name}_summary.wav")

continue_narration = input("Do you want to continue with the full narration? (y/n): ").strip().lower()
if continue_narration == 'y':
    new_pipeline, new_voice = voice_blend.blending(voice_list, weight_list, text_to_narrate)

    # display and save audio segments using method displayed in kokoro documentation:
    for i, (gs, ps, audio) in enumerate(new_pipeline):
        print(f"Blended Voice - Segment {i}:")
        print(i)  # i => index
        print(gs) # gs => graphemes/text
        print(ps) # ps => phonemes
        display(Audio(data=audio, rate=24000))
        sf.write(f'temp\\blended_voice_segment_{i}.wav', audio, 24000) # save each audio file
        

    base_name = os.path.splitext(pdf_to_narrate)[0]
    aj.join_audio_files("temp", narration_name=f"{base_name}.wav") # join audio files
    aj.clear_temp_files() # clear temp files after joining


save_vioce = input("Do you want to save the blended voice? (y/n): ").strip().lower()
if save_vioce == 'y':
    new_voice_name = input("Enter a name for the blended voice (without file extension): ")
    torch.save(new_voice, f'user_voices\\{new_voice_name}.pt')
    print(f'Blended voice saved as user_voices\\{new_voice_name}')  # Save the blended voice tensor
    