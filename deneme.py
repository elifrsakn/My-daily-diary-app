from datasets import load_dataset
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Dataset’i yükle
dataset = load_dataset("go_emotions", "raw")

# 2. DataFrame’e dönüştür
df = pd.DataFrame(dataset["train"])

# 3. Ana duyguyu seç (ilk 1 olan etiketi al)
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
def get_main_emotion(row):
    for emo in emotion_columns:
        if row[emo] == 1:
            return emo
    return "neutral"
df["main_emotion"] = df.apply(get_main_emotion, axis=1)

# 4. LabelEncoder ile sayısal etikete çevir
le = LabelEncoder()
df["label"] = le.fit_transform(df["main_emotion"])

# 5. Basit text temizleme fonksiyonu
def clean_text_simple(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # URL sil
    text = re.sub(r'\@\w+|\#', '', text)                  # Mention ve hashtag sil
    text = re.sub(r'\d+', '', text)                       # Sayı sil
    text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama sil
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text_simple)

# 6. İşlenmiş veriyi Parquet olarak kaydet
df.to_parquet("cleaned_goemotions.parquet", index=False)
print("✅ clean_text ve label eklenmiş DataFrame 'cleaned_goemotions.parquet' dosyasına kaydedildi.")
