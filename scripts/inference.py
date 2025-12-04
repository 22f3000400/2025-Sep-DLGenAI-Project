from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .models import TextCNN


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


def _load_textcnn() -> SeqModelWrapper:
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "artifacts" / "model3_textcnn.pt"

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


def load_all_models() -> Dict[str, SeqModelWrapper]:
    textcnn_wrapper = _load_textcnn()
    return {"TextCNN": textcnn_wrapper}


def predict(
    text: str,
    model_name: str,
    models: Dict[str, SeqModelWrapper],
    threshold: float = 0.5,
) -> Tuple[List[str], np.ndarray, List[str]]:
    wrapper = models[model_name]
    probs = wrapper.predict_proba(text)
    labels = wrapper.label_cols

    active = [lbl for lbl, p in zip(labels, probs) if p >= threshold]
    return labels, probs, active
