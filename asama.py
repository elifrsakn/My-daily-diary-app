import torch
from transformers import AutoModelForSequenceClassification

# 1) Cihazı seç
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Modeli yükle (NUM_LABELS sizin etiket sayınızla aynı olmalı)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS
)

# 3) Modeli GPU/CPU’ye taşı
model.to(device)

# ... bundan sonra train/evaluate kodları ...
