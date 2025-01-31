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

# MacOS Özel Ayarları
load_dotenv()


class Config:
    def __init__(self):
        self.image_folder = "IMG"
        self.translation_model = "Helsinki-NLP/opus-mt-zh-en"
        self.image_to_text_model = "Salesforce/blip-image-captioning-base"
        self.max_text_length = 1000
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.image_types = [".png", ".jpg", ".jpeg", ".heic"]
        self.result_dir = "results"
        self.google_credentials = "google_key.json"
        self.ai_models = {
            "google_gemini": {"enabled": True, "api_key": os.getenv("GOOGLE_GEMINI_API_KEY")},
            "claude_ai": {"enabled": True, "api_key": os.getenv("CLAUDE_AI_API_KEY")},
            "openai": {"enabled": True, "api_key": os.getenv("OPENAI_API_KEY")},
            "huggingface": {"enabled": True, "model": "Salesforce/blip-image-captioning-base"}
        }

    def get_config(self):
        return {
            "image_folder": self.image_folder,
            "translation_model": self.translation_model,
            "image_to_text_model": self.image_to_text_model,
            "max_text_length": self.max_text_length,
            "device": self.device,
            "image_types": self.image_types,
            "result_dir": self.result_dir,
            "google_credentials": self.google_credentials,
            "ai_models": self.ai_models
        }


CONFIG = Config().get_config()


class AIImageProcessor:
    """Farklı AI modellerini kullanarak görsel işleme ve analiz gerçekleştiren sınıf."""

    def __init__(self):
        self.vision_client = self._initialize_vision_client()
        self.caption_pipeline = self._initialize_caption_pipeline()

    def _initialize_vision_client(self):
        """Google Vision client'ı başlat."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                CONFIG['google_credentials']
            )
            return vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as e:
            raise ValueError(f"Google Vision client başlatılamadı: {str(e)}")

    def _initialize_caption_pipeline(self):
        """Görsel açıklama pipeline'ını başlat."""
        try:
            return pipeline(
                "image-to-text",
                model=CONFIG['image_to_text_model'],
                device=CONFIG['device']
            )
        except Exception as e:
            raise ValueError(f"Görsel açıklama pipeline'ı başlatılamadı: {str(e)}")

    def process_image_with_prompt(self, image_path: str) -> dict:
        """Belirli bir prompt ile görseli analiz et."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

            # HEIC formatını JPG'ye dönüştür
            if image_path.lower().endswith(".heic"):
                image_path = self._convert_heic_to_jpg(image_path)

            # OCR çalıştır
            ocr_result = self._run_ocr(image_path)

            # Farklı AI modelleriyle analiz yap
            ai_results = {}
            for model_name, model_config in CONFIG["ai_models"].items():
                if model_config["enabled"]:
                    if model_name == "google_gemini":
                        ai_results[model_name] = self._process_with_google_gemini(image_path, model_config["api_key"])
                    elif model_name == "claude_ai":
                        ai_results[model_name] = self._process_with_claude_ai(image_path, model_config["api_key"])
                    elif model_name == "openai":
                        ai_results[model_name] = self._process_with_openai(image_path, model_config["api_key"])
                    elif model_name == "huggingface":
                        ai_results[model_name] = self._process_with_huggingface(image_path)

            # Prompt formatına uygun çıktı oluştur
            formatted_output = self._format_output(ocr_result, ai_results)

            return formatted_output
        except Exception as e:
            return {"error": f"İşleme hatası: {str(e)}"}

    def _convert_heic_to_jpg(self, heic_path: str) -> str:
        """HEIC formatını JPG'ye çevir."""
        try:
            output_path = os.path.join(
                CONFIG['result_dir'],
                os.path.basename(heic_path).replace(".heic", ".jpg")
            )
            with Image.open(heic_path) as img:
                img.save(output_path, "JPEG")
            return output_path
        except Exception as e:
            raise ValueError(f"HEIC dönüştürme hatası: {str(e)}")

    def _run_ocr(self, image_path: str) -> str:
        """Google Vision OCR çalıştır."""
        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.vision_client.text_detection(
                image=image,
                image_context={"language_hints": ["zh", "en"]}
            )

            if not response.text_annotations:
                return ""

            return response.text_annotations[0].description.strip()
        except Exception as e:
            raise ValueError(f"OCR hatası: {str(e)}")

    def _process_with_google_gemini(self, image_path: str, api_key: str) -> str:
        """Google Gemini ile görsel işle."""
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"
            headers = {"Authorization": f"Bearer {api_key}"}
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Görseli detaylı bir şekilde analiz et."},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                        ]
                    }
                ]
            }
            response = requests.post(url, headers=headers, json=payload)
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:
            raise ValueError(f"Google Gemini hatası: {str(e)}")

    def _process_with_claude_ai(self, image_path: str, api_key: str) -> str:
        """Claude AI ile görsel işle."""
        try:
            url = "https://api.claude.ai/v1/analyze_image"
            headers = {"Authorization": f"Bearer {api_key}"}
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                data = {"prompt": "Görseli detaylı bir şekilde analiz et."}
                response = requests.post(url, headers=headers, files=files, data=data)
            return response.json().get("result", "")
        except Exception as e:
            raise ValueError(f"Claude AI hatası: {str(e)}")

    def _process_with_openai(self, image_path: str, api_key: str) -> str:
        """OpenAI ile görsel işle."""
        try:
            url = "https://api.openai.com/v1/images/generations"
            headers = {"Authorization": f"Bearer {api_key}"}
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            payload = {
                "prompt": "Görseli detaylı bir şekilde analiz et.",
                "image": image_base64
            }
            response = requests.post(url, headers=headers, json=payload)
            return response.json().get("choices", [{}])[0].get("text", "")
        except Exception as e:
            raise ValueError(f"OpenAI hatası: {str(e)}")

    def _process_with_huggingface(self, image_path: str) -> str:
        """HuggingFace ile görsel işle."""
        try:
            result = self.caption_pipeline(image_path)
            return result[0]["generated_text"]
        except Exception as e:
            raise ValueError(f"HuggingFace hatası: {str(e)}")

    def _format_output(self, ocr_result: str, ai_results: dict) -> dict:
        """Çıktıyı belirli bir prompt formatına göre düzenle."""
        try:
            formatted_output = {
                "Görselin Temel Amacı": "Detaylı bir şekilde analiz edilmektedir.",
                "Ürün Tanıtımı (Sol Taraf - 'Pin Chen' Markası)": {
                    "Marka ve Ürün Adı": "Pin Chen",
                    "Görsel": "Açıklama henüz hazır değil.",
                    "Öne Çıkan Özellikler": {}
                },
                "Diğer Markaların Ürünlerinin Tanıtımı (Sağ Taraf - 'Diğer Markalar')": {
                    "Ürün Adı": "Diğer Marka",
                    "Görsel": "Açıklama henüz hazır değil.",
                    "Olumsuz Özellikler": {}
                },
                "Genel Değerlendirme": "Açıklama henüz hazır değil.",
                "Hedef Kitle": "Açıklama henüz hazır değil."
            }

            # OCR ve AI sonuçlarını işleyerek doldur
            if ocr_result:
                formatted_output["Ürün Tanıtımı (Sol Taraf - 'Pin Chen' Markası)"]["Görsel"] = ocr_result

            for model_name, result in ai_results.items():
                formatted_output[f"{model_name} Analizi"] = result

            return formatted_output
        except Exception as e:
            raise ValueError(f"Çıktı düzenleme hatası: {str(e)}")


def initialize_result_directory():
    """Sonuç dizinini oluştur."""
    try:
        result_dir = CONFIG['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            print(f"Sonuç dizini oluşturuldu: {result_dir}")
    except Exception as e:
        raise ValueError(f"Sonuç dizini oluşturma hatası: {str(e)}")


def process_images_in_folder(folder_path: str, processor):
    """Dizindeki tüm görselleri işle."""
    try:
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in CONFIG['image_types']):
                image_path = os.path.join(folder_path, filename)
                print(f"\nProcessing: {image_path}")

                # Görseli işle
                analysis = processor.process_image_with_prompt(image_path)

                # Sonuçları kaydet
                output_filename = f"result_{os.path.splitext(filename)[0]}_analysis.json"
                output_path = os.path.join(CONFIG['result_dir'], output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

                print(f"Sonuçlar kaydedildi: {output_path}")
    except Exception as e:
        raise ValueError(f"Görsel işleme hatası: {str(e)}")


def main():
    print(f"\nMacOS ({platform.machine()}) üzerinde çalışıyor...")

    # Sonuç dizinini oluştur
    initialize_result_directory()

    # Sınıfları başlat
    processor = AIImageProcessor()

    # Dizindeki tüm görselleri işle
    process_images_in_folder(CONFIG['image_folder'], processor)

    print("\nİşlem tamamlandı!")


if __name__ == "__main__":
    main()