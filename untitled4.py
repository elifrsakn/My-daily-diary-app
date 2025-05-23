# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gCinoxULm1wMFASCvapqP_YCQ7RKMdn_
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_parquet('/content/drive/MyDrive/Colab Notebooks/cleaned_goemotions.parquet')
print(df.shape)
df.head()

dist_df = (
    df['main_emotion']
      .value_counts()
      .rename_axis('emotion')
      .reset_index(name='count')
)
dist_df['percent'] = (dist_df['count'] / dist_df['count'].sum() * 100).round(2)
print(dist_df)

df['word_count'] = df['clean_text'].str.split().map(len)
df['char_count'] = df['clean_text'].str.len()

print("\n--- Kelime Sayısı İstatistikleri ---")
print(df['word_count'].describe().round(2))

print("\n--- Karakter Sayısı İstatistikleri ---")
print(df['char_count'].describe().round(2))

import matplotlib.pyplot as plt


# Histogramlar
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].hist(df['word_count'], bins=30)
axes[0].set_title('Kelime Sayısı Dağılımı')
axes[0].set_xlabel('Kelime Sayısı')
axes[0].set_ylabel('Örnek Adedi')

axes[1].hist(df['char_count'], bins=30)
axes[1].set_title('Karakter Sayısı Dağılımı')
axes[1].set_xlabel('Karakter Sayısı')
axes[1].set_ylabel('Örnek Adedi')

# 1. Boş örnekleri kaldır
initial_count = len(df)
df = df[df["word_count"] > 0].reset_index(drop=True)
removed = initial_count - len(df)
print(f"✅ {removed} adet boş örnek silindi. Kalan örnek sayısı: {len(df)}")

# 2. Yeni kelime/karakter istatistikleri
print("\n--- Güncellenmiş Kelime Sayısı İstatistikleri ---")
print(df["word_count"].describe().round(2))

print("\n--- Güncellenmiş Karakter Sayısı İstatistikleri ---")
print(df["char_count"].describe().round(2))

# 3. Yeni sınıf dağılımı
dist_df = (
    df['main_emotion']
      .value_counts()
      .rename_axis('emotion')
      .reset_index(name='count')
)
dist_df['percent'] = (dist_df['count'] / dist_df['count'].sum() * 100).round(2)
print("\n--- Güncellenmiş Duygu Dağılımı ---")
print(dist_df)

# (Opsiyonel) Grafikleri yeniden çizmek istersen
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.bar(dist_df['emotion'], dist_df['count'])
plt.xticks(rotation=45, ha='right')
plt.title('Güncellenmiş Duygu Sınıfı Dağılımı')
plt.ylabel('Örnek Sayısı')
plt.tight_layout()
plt.show()

p95 = df['word_count'].quantile(0.95)
print("95. persentil kelime sayısı:", p95)
df = df[df['word_count'] <= p95]

# 1. 95. persentil üstündekileri filtrele
p95 = df['word_count'].quantile(0.95)
df = df[df['word_count'] <= p95].reset_index(drop=True)
print(f"✅ {p95} kelime üstündeki örnekler atıldı. Kalan satır: {len(df)}")

# 2. Yeni kelime sayısı özetleri
print("\n--- Yeniden Kelime Sayısı İstatistikleri ---")
print(df['word_count'].describe().round(2))

# 3. Yeni duygu dağılımı
dist_df = (
    df['main_emotion']
      .value_counts()
      .rename_axis('emotion')
      .reset_index(name='count')
)
dist_df['percent'] = (dist_df['count'] / dist_df['count'].sum() * 100).round(2)
print("\n--- Yeniden Duygu Dağılımı ---")
print(dist_df)

# 4. (İsteğe bağlı) Grafikleri güncelle
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.bar(dist_df['emotion'], dist_df['count'])
plt.xticks(rotation=45, ha='right')
plt.title('Uçlar Atıldıktan Sonra Duygu Dağılımı')
plt.ylabel('Örnek Sayısı')
plt.tight_layout()
plt.show()

!pip install datasets

# gcsfs’in istediği fsspec sürümüne dön
!pip install fsspec==2025.3.2

!pip install --upgrade \
  fsspec==2025.3.0 gcsfs==2025.3.0 \
  && pip install --upgrade \
  torch torchvision --upgrade-strategy only-if-needed

import pandas as pd
from datasets import Dataset  # <- Burada Dataset sınıfını alıyoruz

hf_ds = Dataset.from_pandas(
    df[['clean_text','label']].rename(columns={'clean_text':'text'})
).shuffle(seed=42)

from datasets import Dataset, DatasetDict

# hf_ds senin tüm veri setin
# %80 eğitim, %20 doğrulama
split = hf_ds.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

print(f"✅ Train örnek sayısı: {len(train_ds)}")
print(f"✅ Val   örnek sayısı: {len(val_ds)}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# remove_columns ile 'text' sütununu atıyoruz, geriye token id’ler kalacak
train_tok = train_ds.map(tokenize_batch, batched=True, batch_size=500, remove_columns=["text"])
val_tok   = val_ds.map(  tokenize_batch, batched=True, batch_size=500, remove_columns=["text"])

print("✅ Tokenization tamamlandı. Örnek:\n", train_tok[0])

# Eğer hâlen pandas DataFrame üzerinde çalışıyorsanız:
NUM_LABELS = df['label'].nunique()
print("🔢 Kaç sınıf var:", NUM_LABELS)

# Ya da HF Dataset kullanıyorsanız:
# NUM_LABELS = train_tok.features['label'].num_classes

from transformers import TrainingArguments, IntervalStrategy

# 1) TrainingArguments oluştur
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    logging_dir="./logs",
    report_to="wandb",   # ya da "none" diyebilirsin
)

# 2) Değerlendirme stratejisini ayarla (epoch sonunda, delay=0 ile)
training_args = training_args.set_evaluate(
    strategy=IntervalStrategy.EPOCH,  # veya "epoch"
    delay=0.0,                        # 0 epoch beklemeden hemen her epoch incele
    batch_size=16,                    # eval batch size
    # steps parametresi sadece strategy="steps" için geçerli
)

# 3) Trainer’ı başlat
from transformers import Trainer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
)

# 4) Eğitimi başlat
trainer.train()

from transformers import TrainingArguments, IntervalStrategy, Trainer, AutoModelForSequenceClassification
import torch, os, shutil

# 0) Kaç sınıf varsa belirle
NUM_LABELS = train_tok.features['label'].num_classes

# 1) TrainingArguments oluştur
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    logging_dir="./logs",
    report_to="wandb",   # veya "none"
)
training_args = training_args.set_evaluate(
    strategy=IntervalStrategy.EPOCH,  # her epoch sonunda validasyon
    delay=0.0,                        # hemen ilk epoch sonunda da çalıştır
    batch_size=16,                    # eval batch size
)

# 2) Model + Trainer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
)

# 3) Eski checkpoint klasörlerini temizle
output_dir = training_args.output_dir
if os.path.isdir(output_dir):
    removed = 0
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint"):
            shutil.rmtree(os.path.join(output_dir, name))
            removed += 1
    print(f"✅ {removed} adet eski checkpoint silindi.")
else:
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ '{output_dir}' dizini oluşturuldu.")

# 4) Eğitimi kaldığın yerden devam ettir
trainer.train(resume_from_checkpoint=True)

!pip install -q --upgrade transformers