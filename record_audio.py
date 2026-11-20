
import sounddevice as sd
import soundfile as sf

duration = 10
sample_rate = 16000

print("Recording started! Speak now...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(
    "dataset/audio/test.wav",
    audio,
    sample_rate
)

print("Recording saved successfully!")