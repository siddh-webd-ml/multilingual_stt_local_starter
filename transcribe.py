
import os
import warnings
import logging

# Force completely offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress routine warnings and logs
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import argparse
from pathlib import Path

import torch
import librosa
import soundfile as sf

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from transformers.utils import logging as hf_logging
from peft import PeftModel

hf_logging.set_verbosity_error()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--language", default="english")

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    base = "openai/whisper-small"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load processor from local cache or adapter
    processor_path = args.adapter if args.adapter else base

    processor = WhisperProcessor.from_pretrained(
        processor_path,
        local_files_only=True
    )

    # Load Whisper model locally
    model = WhisperForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=dtype,
        local_files_only=True
    )

    # Load optional LoRA adapter
    if args.adapter:
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=True
        )

    model.to(device)
    model.eval()

    # Read audio
    audio, sr = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False
    )

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz
    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

    # Prepare input features
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(
        device=device,
        dtype=dtype
    )

    # Clear legacy forced decoder settings
    model.generation_config.forced_decoder_ids = None

    # Generate transcription
    with torch.inference_mode():

        predicted_ids = model.generate(
            input_features,
            language=args.language,
            task="transcribe"
        )

    # Decode transcription
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    # ONLY output
    print(transcription.strip())


if __name__ == "__main__":
    main()
