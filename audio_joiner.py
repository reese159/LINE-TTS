from pydub import AudioSegment # type: ignore
import os
import torch
import numpy as np

def join_audio_files(temp_path, narration_name="user_narration.wav"):
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
        


def add_audio_to_narration(temp_path=NotImplemented, narration_name="user_narration.wav", new_chunk=None):
    # Initialize an empty AudioSegment to store the combined audio
    combined_audio = AudioSegment.empty()

    # Simulate a loop where you get audio chunks
    for i in range(5):
        # In a real scenario, 'new_chunk' would come from
        # a recording, a file, or another processing step.
        # Here, we create a silent chunk for demonstration.
        new_chunk = AudioSegment.silent(duration=100)  # testing .1 second of silence

        # Append the new chunk to the combined audio
        combined_audio += new_chunk

    # Export the final combined audio to a file
    combined_audio.export(narration_name, format="wav")

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