from pydub import AudioSegment # type: ignore

# Load audio files
audio1 = AudioSegment.from_file("0.wav", format="wav")
audio2 = AudioSegment.from_file("1.wav", format="wav")

# Combine audio files (concatenation)
combined_audio = audio1 + audio2

# Export the combined audio
combined_audio.export("combined_output.wav", format="wav")

print("Audio files combined successfully!")