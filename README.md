# Local English Speech-to-Text

A locally running English Speech-to-Text (STT) system powered by OpenAI Whisper Small and Hugging Face Transformers.

The project converts spoken English audio into written text using a locally downloaded AI model. Once the model and dependencies are installed, transcription can run completely offline without sending audio to external servers or cloud APIs.

## Features

- **Local transcription:** Process audio directly on your computer.
- **Offline inference:** No internet connection required after the initial model and dependency setup.
- **English recognition:** Designed for English speech transcription.
- **GPU acceleration:** Supports NVIDIA CUDA when a compatible GPU and PyTorch installation are available.
- **CPU support:** Can also run without an NVIDIA GPU.
- **Audio preprocessing:** Converts stereo audio to mono and resamples it to 16 kHz.
- **LoRA adapter support:** Supports loading a trained PEFT/LoRA adapter.
- **Evaluation-ready:** Includes dependencies for Word Error Rate (WER) evaluation and future model fine-tuning.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| OpenAI Whisper Small | Speech recognition model |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Model loading and inference |
| PEFT | LoRA fine-tuning |
| Librosa | Audio preprocessing |
| SoundFile | Audio file reading |
| NumPy & Pandas | Data processing |
| JiWER | Word Error Rate evaluation |

## Project Structure

```text
multilingual_stt_local_starter/
│
├── dataset/
│   ├── audio/
│   │   └── README.txt
│   ├── train.csv
│   └── validation.csv
│
├── check_gpu.py
├── transcribe.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3.11 recommended
- Windows, Linux, or macOS
- NVIDIA GPU with compatible CUDA support (optional)
- Internet connection for initial setup and model download
- Sufficient storage for model weights, Python packages, and audio files

The project can run on CPU, but GPU acceleration is recommended for faster inference and training.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the repository URL and directory name with your actual GitHub repository details.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install PyTorch

Install PyTorch separately according to your hardware.

Visit the official installation selector:

https://pytorch.org/get-started/locally/

Choose:
- CPU if you don't have a compatible NVIDIA GPU.
- The appropriate CUDA build if you have a compatible NVIDIA GPU.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Whisper Small

The model must be downloaded once before offline use.

Run:

```bash
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-small'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-small'); print('Whisper Small downloaded successfully!')"
```

This downloads the pretrained model and processor into the local Hugging Face cache.

**Important:** Keep the downloaded model cache available. The inference script uses local-only loading and will not download missing model files.

## Usage

### 1. Prepare an audio file

Supported input formats depend on SoundFile's installed audio backends. WAV is recommended.

Recommended audio format:

- Sample rate: 16,000 Hz
- Channels: Mono
- Format: WAV
- Language: English

Place your audio file inside:

```text
dataset/audio/
```

### 2. Run transcription

```bash
python transcribe.py --audio dataset/audio/test.wav
```

The model will:
1. Load Whisper Small from the local cache.
2. Read the supplied audio file.
3. Convert stereo audio to mono if necessary.
4. Resample the audio to 16 kHz if needed.
5. Generate the English transcription.
6. Print only the recognized text.

### 3. Specify the language

English is the default language.

You can also explicitly specify it:

```bash
python transcribe.py --audio dataset/audio/test.wav --language english
```

### 4. Use a trained LoRA adapter

If you have trained a compatible LoRA adapter, provide its directory:

```bash
python transcribe.py --audio dataset/audio/test.wav --adapter ./output
```

The adapter must contain the required PEFT files. The base Whisper Small model must also be available locally.

## Offline Operation

Offline inference is a core requirement of this project.

The transcription script enables:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

Model loading also uses:

```python
local_files_only=True
```

This ensures that inference uses locally available model files instead of requesting them from Hugging Face.

**Initial setup requires internet access** to install dependencies and download Whisper Small. After that, inference can run without internet.

No cloud transcription API is required.

## Model Information

**Model:** `openai/whisper-small`

Whisper Small is a pretrained multilingual speech recognition model. This project uses its English transcription capability.

The current implementation uses the pretrained model as its baseline. Fine-tuning with a suitable English speech dataset can be explored to improve recognition for the target accents and recording conditions.

## Evaluation

The project includes `evaluate` and `jiwer` dependencies for evaluating transcription quality.

The primary evaluation metric is Word Error Rate (WER).

WER measures transcription errors through:

- Substitutions
- Deletions
- Insertions

A lower WER indicates better transcription accuracy.

## Current Limitations

- Recognition accuracy depends on audio quality, accent, pronunciation, and background noise.
- The current inference script processes a single Whisper input window; longer recordings require chunking.
- English is the current project focus.
- GPU performance depends on the hardware and available memory.
- The pretrained baseline has not been fine-tuned for every English accent.

## Privacy

Audio is processed locally by the inference script.

Once the model and dependencies are installed, audio does not need to be uploaded to a third-party transcription service.

## Future Improvements

- Fine-tuning Whisper Small with English speech datasets.
- Improving recognition of diverse English accents.
- Supporting longer audio recordings through chunking.
- Evaluating model performance using WER.
- Optimizing inference speed and memory consumption.
- Improving robustness in noisy environments.

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

Add an appropriate license before distributing this project. Check the licenses of the pretrained model and all datasets used.

---

**Built with Python, PyTorch, and OpenAI Whisper Small.**