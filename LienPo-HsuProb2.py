import whisper

def main():
    # Load the tiny.en model
    model = whisper.load_model("tiny.en")

    # Transcribe the audio file; fp16=False avoids errors on CPU-only systems
    result = model.transcribe("./MLKDreamShort.wav", fp16=False)

    # Print the transcription result
    print("Transcription result:")
    print(result["text"])

if __name__ == "__main__":
    main()
