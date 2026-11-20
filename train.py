import os
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import evaluate

from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
BASE_MODEL = os.getenv("STT_BASE_MODEL", "openai/whisper-small")
EPOCHS = int(os.getenv("STT_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("STT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("STT_GRAD_ACCUM", "8"))
LR = float(os.getenv("STT_LR", "1e-4"))
MAX_AUDIO_SECONDS = float(os.getenv("STT_MAX_AUDIO_SECONDS", "30"))
SR = 16000

def read_manifest(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"audio", "text", "language"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must have columns: audio,text,language")
        for row in reader:
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            text = row["text"].strip()
            language = row["language"].strip().lower()
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            if not text or not language:
                raise ValueError(f"Empty transcript/language in {csv_path}: {row}")
            rows.append({"audio": audio_path, "text": text, "language": language})
    if not rows:
        raise ValueError(f"No rows found in {csv_path}. Add audio/transcript rows first.")
    return rows

class SpeechDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio, sr = sf.read(str(row["audio"]), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if len(audio) == 0:
            raise ValueError(f"Empty audio: {row['audio']}")
        if len(audio) / SR > MAX_AUDIO_SECONDS:
            audio = audio[: int(MAX_AUDIO_SECONDS * SR)]

        features = self.processor.feature_extractor(
            audio, sampling_rate=SR, return_tensors="np"
        ).input_features[0]

        # Whisper needs the language/task prefix for each training example.
        self.processor.tokenizer.set_prefix_tokens(
            language=row["language"], task="transcribe"
        )
        labels = self.processor.tokenizer(row["text"]).input_ids

        return {"input_features": features, "labels": labels}

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        x = torch.tensor(
            np.stack([f["input_features"] for f in features]),
            dtype=torch.float32
        )
        label_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        # The decoder start token is supplied by the model, not repeated in labels.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        return {"input_features": x, "labels": labels}

def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This will be very slow on CPU.")
    else:
        print("Training GPU:", torch.cuda.get_device_name(0))
        print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    train_rows = read_manifest(DATA_DIR / "train.csv")
    val_rows = read_manifest(DATA_DIR / "validation.csv")
    print(f"Train rows: {len(train_rows)} | Validation rows: {len(val_rows)}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    train_ds = SpeechDataset(train_rows, processor)
    val_ds = SpeechDataset(val_rows, processor)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = "transcribe"

    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_text, references=label_text)}

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting fine-tuning...")
    trainer.train()

    adapter_dir = OUT_DIR / "lora_adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Done. LoRA adapter saved to: {adapter_dir}")
    print("Use transcribe.py --adapter outputs/lora_adapter to test it.")

if __name__ == "__main__":
    main()
