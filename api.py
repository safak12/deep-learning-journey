from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="Turkce Haber Siniflandirici", version="1.0")

print("Model yukleniyor...")
tokenizer = AutoTokenizer.from_pretrained("model/turkce_haber_tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("model/turkce_haber_model")
model.eval()

kategoriler = {0: "Spor", 1: "Ekonomi", 2: "Teknoloji", 3: "Siyaset", 4: "Saglik"}

class HaberInput(BaseModel):
    metin: str

class TahminOutput(BaseModel):
    metin: str
    kategori: str
    guven: float
    tum_olasiliklar: dict

@app.get("/")
def ana_sayfa():
    return {"mesaj": "Turkce Haber Siniflandirici API", "durum": "aktif"}

@app.post("/predict")
def tahmin_et(haber: HaberInput):
    encoding = tokenizer(
        haber.metin,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
    probs = torch.softmax(output.logits, dim=1)[0]
    pred = probs.argmax().item()

    return {
        "metin": haber.metin,
        "kategori": kategoriler[pred],
        "guven": round(probs[pred].item(), 4),
        "tum_olasiliklar": {v: round(probs[i].item(), 4) for i, v in kategoriler.items()}
    }

@app.get("/health")
def saglik_kontrolu():
    return {"durum": "saglikli"}