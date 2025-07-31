from pydub import AudioSegment # type: ignore
import os
import torch
import numpy as np

def join_audio_files(temp_path, narration_name="user_narration.wav"):
    '''
    :temp_path: Path to the directory containing temporary audio files
    :narration_name: Name of the final audio file to be created
    This function combines all .wav files in the specified directory into a single audio file.
    '''
    combined_audio = AudioSegment.empty()
    
    for filename in os.listdir(temp_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(temp_path, filename)
            audio_segment = AudioSegment.from_wav(file_path)
            combined_audio += audio_segment

    try:
        # export combined audio
        combined_audio.export(f"final_narrations\\{narration_name}", format="wav")
        print(f"Audio files combined successfully into {narration_name}")
    except Exception as e:
        print(f"Error exporting combined audio: {e}")
        

def clear_temp_files():
    import os
    temp_dir = 'temp'
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    


def tensor_to_audio_segment(audio_tensor, sample_rate=24000):
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.squeeze().cpu().numpy()
    else:
        audio_np = audio_tensor

    audio_np = (audio_np * 32767).astype(np.int16)  # convert to 16-bit PCM
    audio_segment = AudioSegment(
        audio_np.tobytes(), 
        frame_rate=sample_rate,
        sample_width=2,  # 2 bytes for 16-bit
        channels=1
    )
    return audio_segment