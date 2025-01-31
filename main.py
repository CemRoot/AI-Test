"""
MacOS Optimize Edilmiş OCR ve Çeviri Sistemi

Önemli MacOS Ayarları:
- Homebrew ile gerekli kütüphanelerin kurulumu
- Python sanal ortam desteği
- Apple Silicon (M1/M2) GPU desteği
"""

import os
import json
import base64
import time
import torch
import platform
from google.cloud import vision_v1 as vision
from google.oauth2 import service_account
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
import requests
from PIL import Image
from dotenv import load_dotenv

# MacOS Özel Konfigürasyon
load_dotenv()
CONFIG = {
    "google_credentials": "google_key.json",
    "image_folder": "IMG",
    "translation_model": "Helsinki-NLP/opus-mt-zh-en",
    "image_to_text_model": "Salesforce/blip-image-captioning-base",
    "max_text_length": 1000,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "image_types": [".png", ".jpg", ".jpeg", ".heic"],
    "temp_dir": "/tmp/ocr_processing"
}

# MacOS için Ön Hazırlık
if not os.path.exists(CONFIG['temp_dir']):
    os.makedirs(CONFIG['temp_dir'], exist_ok=True)


class MacOCRProcessor:
    """MacOS Optimize Edilmiş OCR İşlemcisi"""

    def __init__(self):
        # Google Vision Client
        self.vision_client = vision.ImageAnnotatorClient(
            credentials=service_account.Credentials.from_service_account_file(
                CONFIG['google_credentials']
            )
        )

        # Görsel İşleme Pipeline
        self.caption_pipeline = pipeline(
            "image-to-text",
            model=CONFIG['image_to_text_model'],
            device=CONFIG['device']
        )

    def process_image(self, image_path: str) -> dict:
        """Görseli MacOS-friendly formatında işle"""
        try:
            # HEIC formatı desteği
            if image_path.lower().endswith(".heic"):
                image_path = self._convert_heic_to_jpg(image_path)

            return self._analyze_image(image_path)
        except Exception as e:
            return {"error": f"MacOS processing error: {str(e)}"}

    def _convert_heic_to_jpg(self, heic_path: str) -> str:
        """HEIC'i JPG'e çevir (MacOS özelliği)"""
        jpg_path = os.path.join(CONFIG['temp_dir'], os.path.basename(heic_path).replace(".heic", ".jpg"))
        img = Image.open(heic_path)
        img.save(jpg_path, "JPEG")
        return jpg_path

    def _analyze_image(self, image_path: str) -> dict:
        """Görsel analizi yap"""
        result = {"ocr": None, "caption": None}

        # OCR İşlemi
        ocr_result = self._run_google_ocr(image_path)
        if ocr_result["success"]:
            result["ocr"] = ocr_result["text"]

        # Görsel Açıklama
        caption_result = self.caption_pipeline(image_path)
        result["caption"] = caption_result[0]["generated_text"]

        return result

    def _run_google_ocr(self, image_path: str) -> dict:
        """Google Cloud Vision OCR"""
        try:
            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = self.vision_client.text_detection(
                image=image,
                image_context={"language_hints": ["zh", "en"]}
            )

            texts = [annotation.description for annotation in response.text_annotations]
            return {
                "success": True,
                "text": "\n".join(texts[1:]) if texts else ""  # İlk eleman tüm metni içeriyor
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class MacTranslator:
    """Apple Silicon Optimize Çevirmen"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['translation_model'])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            CONFIG['translation_model']
        ).to(CONFIG['device'])

    def translate(self, text: str) -> str:
        """Metni çevir ve MacOS optimizasyonu uygula"""
        try:
            inputs = self.tokenizer(
                text[:CONFIG['max_text_length']],
                return_tensors="pt",
                truncation=True
            ).to(CONFIG['device'])

            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Translation error: {str(e)}"


def main():
    print(f"MacOS ({platform.machine()}) üzerinde çalışıyor...")

    processor = MacOCRProcessor()
    translator = MacTranslator()

    for filename in os.listdir(CONFIG['image_folder']):
        if any(filename.lower().endswith(ext) for ext in CONFIG['image_types']):
            image_path = os.path.join(CONFIG['image_folder'], filename)
            print(f"\nProcessing: {image_path}")

            # Görsel İşleme
            analysis = processor.process_image(image_path)

            # Çeviri
            if analysis.get("ocr"):
                translated = translator.translate(analysis["ocr"])
                analysis["translation"] = translated

            # Sonuçları Kaydet
            output_file = os.path.join(CONFIG['temp_dir'], f"result_{filename}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            print(f"Sonuçlar kaydedildi: {output_file}")


if __name__ == "__main__":
    main()