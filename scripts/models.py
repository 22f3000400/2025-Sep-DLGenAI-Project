import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        filter_sizes,
        num_labels: int,
        padding_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        conv_outs = []
        for conv in self.convs:
            h = conv(x)
            h = F.relu(h)
            h = F.max_pool1d(h, h.shape[-1])
            conv_outs.append(h.squeeze(-1))

        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


class EmotionBERT(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits
