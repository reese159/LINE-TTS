from pydub import AudioSegment # type: ignore
import os

def join_audio_files(temp_path, output_file, narration_name="user_narration.wav"):
    combined_audio = AudioSegment.empty()
    
    for filename in os.listdir(temp_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(temp_path, filename)
            try:
                audio_segment = AudioSegment.from_wav(file_path)
                if combined_audio is None:
                    combined_audio = audio_segment
                else:
                    combined_audio += audio_segment
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

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