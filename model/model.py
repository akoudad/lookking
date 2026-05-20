import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
LABEL_NAMES = ["Low Match", "Medium Match", "High Match"]
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128


class LookkingMatcher(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)

        # Freeze embeddings + first 4 of 6 transformer layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.drop(cls))


def get_tokenizer() -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
