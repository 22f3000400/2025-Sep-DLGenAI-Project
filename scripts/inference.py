from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import BertTokenizer

from .models import TextCNN, EmotionBERT
import os
import streamlit as st

if "KAGGLE_USERNAME" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

try:
    import kagglehub
except ImportError:
    kagglehub = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = ["fear", "sadness", "surprise", "joy", "anger"]


def preprocess_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def encode_text(text: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    tokens = preprocess_text(text).split()
    unk_id = stoi.get("<unk>", 1)
    pad_id = stoi.get("<pad>", 0)
    ids = [stoi.get(tok, unk_id) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))
    return ids


def _resolve_ckpt_path(dataset_slug: str, filename: str, local_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    local_path = project_root / "artifacts" / local_name
    if local_path.exists():
        return local_path

    if kagglehub is None:
        raise FileNotFoundError(
            f"Checkpoint {local_name} not found locally and kagglehub is not installed."
        )

    ds_path = kagglehub.dataset_download(dataset_slug)
    hub_path = Path(ds_path) / filename
    if not hub_path.exists():
        raise FileNotFoundError(
            f"File {filename} not found in Kaggle dataset {dataset_slug}"
        )
    return hub_path


class SeqModelWrapper:
    def __init__(self, model: torch.nn.Module, stoi, max_len: int, label_cols: List[str]):
        self.model = model
        self.stoi = stoi
        self.max_len = max_len
        self.label_cols = list(label_cols)

    def predict_proba(self, text: str) -> np.ndarray:
        ids = encode_text(text, self.stoi, self.max_len)
        x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs


class BertModelWrapper:
    def __init__(
        self,
        model: EmotionBERT,
        tokenizer: BertTokenizer,
        max_len: int,
        label_cols: List[str],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = list(label_cols)

    def predict_proba(self, text: str) -> np.ndarray:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs


def _load_textcnn() -> SeqModelWrapper:
    ckpt_path = _resolve_ckpt_path(
        dataset_slug="sramakrishnan2904/model3-textcnn-checkpoint",
        filename="model3_textcnn.pt",
        local_name="model3_textcnn.pt",
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    vocab_size = ckpt["vocab_size"]
    embed_dim = ckpt["embed_dim"]
    num_filters = ckpt["num_filters"]
    filter_sizes = ckpt["filter_sizes"]
    label_cols = ckpt["label_cols"]
    stoi = ckpt["stoi"]
    max_len = ckpt["max_len"]

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_labels=len(label_cols),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return SeqModelWrapper(model, stoi, max_len, label_cols)


def _load_bert() -> BertModelWrapper:
    ckpt_path = _resolve_ckpt_path(
        dataset_slug="sramakrishnan2904/model5-bert-checkpoint",
        filename="model5_bert_checkpoint.pt",
        local_name="model5_bert_checkpoint.pt",
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hf_model_name = ckpt["hf_model_name"]
    max_len = ckpt["max_len"]
    label_cols = ckpt["label_cols"]

    tokenizer = BertTokenizer.from_pretrained(hf_model_name)
    model = EmotionBERT(hf_model_name, num_labels=len(label_cols)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return BertModelWrapper(model, tokenizer, max_len, label_cols)


def load_all_models() -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["TextCNN"] = _load_textcnn()
    try:
        models["BERT"] = _load_bert()
    except Exception as e:
        print(f"Warning: could not load BERT model: {e}")
    return models


def predict(
    text: str,
    model_name: str,
    models: Dict[str, object],
    threshold: float = 0.5,
) -> Tuple[List[str], np.ndarray, List[str]]:
    wrapper = models[model_name]
    probs = wrapper.predict_proba(text)
    labels = wrapper.label_cols
    active = [lbl for lbl, p in zip(labels, probs) if p >= threshold]
    return labels, probs, active
