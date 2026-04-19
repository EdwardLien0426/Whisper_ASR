import whisper
import speech_recognition as sr
import numpy as np
import librosa
import time

# =========================
# Configuration
# =========================
MODEL_SIZE = "tiny.en"      # Whisper model size
TARGET_SR = 16000           # Required sample rate for Whisper
CHUNK_SECONDS = 2           # Recording duration per chunk (seconds)
BUFFER_SECONDS = 15         # Rolling audio buffer length (seconds)
DECODE_INTERVAL = 2         # Transcription interval (seconds)

# =========================
# Main Function
# =========================
def main():
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE)

    recognizer = sr.Recognizer()

    # List all available microphones
    print("Available microphones:")
    mic_list = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_list):
        print(f"{i}: {name}")

    mic_index = int(input("Enter microphone index: "))

    # Rolling audio buffer (stored entirely in memory)
    audio_buffer = np.zeros(0, dtype=np.float32)

    last_decode_time = 0
    last_text = ""

    with sr.Microphone(device_index=mic_index) as source:
        # Most microphones use a 44.1 kHz sampling rate
        source.SAMPLE_RATE = 44100

        print("Calibrating ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Start speaking (Press Ctrl+C to stop)")

        try:
            while True:
                # Record a short audio segment from the microphone
                audio = recognizer.listen(
                    source,
                    phrase_time_limit=CHUNK_SECONDS,
                    timeout=None
                )

                # Convert raw audio bytes to a float32 numpy array
                raw_audio = audio.get_raw_data()
                wav = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
                wav /= 32768.0

                # Resample the audio to 16 kHz for Whisper
                wav = librosa.resample(
                    wav,
                    orig_sr=source.SAMPLE_RATE,
                    target_sr=TARGET_SR
                )

                # Append the new audio to the rolling buffer
                audio_buffer = np.concatenate([audio_buffer, wav])

                # Keep only the most recent BUFFER_SECONDS of audio
                max_samples = BUFFER_SECONDS * TARGET_SR
                if len(audio_buffer) > max_samples:
                    audio_buffer = audio_buffer[-max_samples:]

                # Run transcription every DECODE_INTERVAL seconds
                now = time.time()
                if now - last_decode_time >= DECODE_INTERVAL:
                    last_decode_time = now

                    result = model.transcribe(
                        audio_buffer,
                        language="en",
                        fp16=False,
                        temperature=0.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False
                    )

                    text = result["text"].strip()

                    # Print only when the transcription result changes
                    if text and text != last_text:
                        print(">>", text)
                        last_text = text

        except KeyboardInterrupt:
            print("\nStopped by user.")

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
