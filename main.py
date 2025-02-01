import os
import json
import time
import base64
import logging
import requests
from datetime import datetime

import torch
from google.cloud import vision_v1 as vision
from google.oauth2 import service_account
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
from dotenv import load_dotenv

# Ortam değişkenlerini .env dosyasından yükle (opsiyonel)
load_dotenv()


########################################
# 1. CONFIG YÖNETİCİSİ: config.json OKUMA
########################################

class ConfigManager:
    def __init__(self, config_path="config.json"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"[INFO] {config_path} başarıyla yüklendi.")
        except Exception as e:
            print(f"[ERROR] {config_path} okunamadı: {e}")
            self.config = {}

        # Genel dizin ayarları
        general = self.config.get("general", {})
        self.img_folder = general.get("img_folder", "IMG")
        self.results_folder = general.get("results_folder", "results")
        self.log_folder = general.get("log_folder", "logs")
        self.allowed_extensions = general.get("allowed_extensions", [".jpg", ".jpeg", ".png"])
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

    def get(self):
        return self.config


########################################
# 2. LOG YÖNETİMİ
########################################

def setup_logging(log_folder):
    log_filename = os.path.join(log_folder, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Terminale de logları yazdırmak için:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(f"Log dosyası: {log_filename}")


########################################
# 3. MALİYET YÖNETİCİSİ (CostEvaluator)
########################################

class CostEvaluator:
    def __init__(self, config):
        # Config dosyanızda "cost" bölümü şu şekilde olabilir:
        # "cost": {
        #    "gemini": 0.001,
        #    "openai": 0.001,
        #    "anthropic": 0.001,
        #    "huggingface": 0.001,
        #    "ocr": 0.0005,
        #    "translation": 0.0003
        # }
        self.cost_config = config.get("cost", {})

    def evaluate_cost(self, model_name, response):
        # Manuel olarak config dosyasındaki değeri döndürür.
        return self.cost_config.get(model_name.lower(), 0.001)


########################################
# 4. PROMPT PIPELINE (Gemini, OpenAI, Anthropic, HuggingFace)
########################################

# Global tokenizer (gpt2) ile token sayısını daha doğru hesaplayalım.
tokenizer_for_count = AutoTokenizer.from_pretrained("gpt2")


class PromptPipeline:
    def __init__(self, config, cost_evaluator):
        # Config dosyasındaki "models" bölümünü alıyoruz.
        # Örnek: {"gemini": { ... }, "openai": { ... }, "anthropic": { ... }, "huggingface": { ... }}
        self.config = config.get("models", {})
        self.general_prompt = "Lütfen görseli analiz edip aşağıdaki kriterlere göre yorum yapınız:"  # Standart prompt
        self.cost_evaluator = cost_evaluator
        # HuggingFace için pipeline instance'ı; burada lazy initialization da yapılabilir.
        self.hf_pipeline = {}

    def process_image(self, image_path, user_prompt=None):
        results = {}
        # Resmi base64'e çeviriyoruz (API çağrıları için gereklidir).
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logging.info(f"Resim {os.path.basename(image_path)} base64'e çevrildi.")
        except Exception as e:
            logging.error(f"Resim okunamadı: {e}")
            return results

        # Kullanıcı prompt'u varsa onu, yoksa standart prompt kullanılır.
        prompt = user_prompt if user_prompt else self.general_prompt

        # Tüm modeller için aynı prompt gönderilecek.
        for model_name, settings in self.config.items():
            if not settings.get("enabled", False):
                logging.info(f"{model_name} modeli devre dışı bırakılmış, atlanıyor.")
                continue

            # Gemin, OpenAI, Anthropic için API çağrısı yapılıyor.
            if model_name.lower() in ["gemini", "openai", "anthropic"]:
                api_key = settings.get("api_key")
                endpoint = settings.get("endpoint")
                prompt_template = settings.get("prompt_template", prompt)
                if api_key is None or endpoint is None:
                    logging.info(f"{model_name} için api_key veya endpoint ayarı boş, atlanıyor.")
                    continue

                logging.info(f"{model_name} için API çağrısı başlatılıyor.")
                start_time = time.time()
                try:
                    if model_name.lower() == "gemini":
                        response = self.call_gemini_api(endpoint, api_key, prompt_template, image_base64)
                    elif model_name.lower() == "openai":
                        response = self.call_openai_api(endpoint, api_key, prompt_template, image_base64)
                    elif model_name.lower() == "anthropic":
                        response = self.call_anthropic_api(endpoint, api_key, prompt_template, image_base64)
                except Exception as e:
                    logging.error(f"{model_name} API çağrısı sırasında hata: {e}")
                    response = {}
                elapsed = time.time() - start_time
                cost = self.cost_evaluator.evaluate_cost(model_name, response)
                token_size = self.evaluate_token_size(response)
                results[model_name] = {
                    "response": response,
                    "time_spent_sec": elapsed,
                    "cost": cost,
                    "token_size": token_size
                }
                logging.info(f"{model_name}: Süre = {elapsed:.2f} s, Maliyet = {cost}, Token sayısı = {token_size}")

            # HuggingFace için yerel inference (Transformers pipeline) kullanılıyor.
            elif model_name.lower() == "huggingface":
                try:
                    hf_model = settings.get("model")
                    if hf_model is None:
                        logging.info("HuggingFace için model adı belirtilmemiş, atlanıyor.")
                        continue

                    # Pipeline instance'ı yoksa oluştur.
                    if model_name.lower() not in self.hf_pipeline:
                        self.hf_pipeline[model_name.lower()] = pipeline("image-to-text", model=hf_model)

                    start_time = time.time()
                    # HuggingFace pipeline, genellikle PIL image nesnesi bekler.
                    image = Image.open(image_path)
                    hf_response = self.hf_pipeline[model_name.lower()](image)
                    elapsed = time.time() - start_time
                    cost = self.cost_evaluator.evaluate_cost(model_name, hf_response)
                    # Pipeline çıktısında "generated_text" veya "caption" anahtarı olabilir.
                    generated_text = hf_response[0].get("generated_text", hf_response[0].get("caption", ""))
                    token_size = self.count_tokens(generated_text)
                    results[model_name] = {
                        "response": hf_response,
                        "time_spent_sec": elapsed,
                        "cost": cost,
                        "token_size": token_size
                    }
                    logging.info(f"{model_name}: Süre = {elapsed:.2f} s, Maliyet = {cost}, Token sayısı = {token_size}")
                except Exception as e:
                    logging.error(f"HuggingFace işlemi sırasında hata: {e}")
                    results[model_name] = {"response": {"error": str(e)}}
            else:
                logging.info(f"{model_name} için tanımlı bir API çağrısı yok, atlanıyor.")
        return results

    def call_gemini_api(self, endpoint, api_key, prompt_template, image_base64):
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_template},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                ]
            }]
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        return response.json()

    def call_openai_api(self, endpoint, api_key, prompt_template, image_base64):
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        return response.json()

    def call_anthropic_api(self, endpoint, api_key, prompt_template, image_base64):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                    {"type": "text", "text": prompt_template}
                ]
            }]
        }
        response = requests.post(endpoint, headers=headers, json=payload)
        return response.json()

    def evaluate_token_size(self, response):
        text = ""
        # Gemini yanıtı
        if "candidates" in response:
            text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        # OpenAI yanıtı
        elif "choices" in response:
            text = response["choices"][0]["message"].get("content", "")
        # Anthropic yanıtı
        elif "content" in response:
            # Anthropic yanıtında "content" listesinde metin bulunur.
            text = response["content"][0].get("text", "")
        return self.count_tokens(text)

    def count_tokens(self, text):
        if text:
            tokens = tokenizer_for_count.encode(text)
            return len(tokens)
        return 0


########################################
# 5. OCR & ÇEVİRİ PIPELINE (Google Vision OCR – HuggingFace Çeviri)
########################################

class OCRTranslationPipeline:
    def __init__(self, config, cost_evaluator):
        ocr_config = config.get("ocr", {}).get("google", {})
        self.ocr_enabled = ocr_config.get("enabled", False)
        self.google_credentials = ocr_config.get("credentials", None)  # Örneğin "google_key.json"
        self.language_hints = ocr_config.get("language_hints", ["en"])

        translation_config = config.get("translation", {})
        self.translation_enabled = translation_config.get("enabled", False)
        self.translation_model = translation_config.get("model", None)

        self.cost_evaluator = cost_evaluator

        # Google Vision Client – credentials sağlanmışsa
        if self.ocr_enabled and self.google_credentials:
            try:
                credentials = service_account.Credentials.from_service_account_file(self.google_credentials)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                logging.info("Google Vision Client başarıyla başlatıldı.")
            except Exception as e:
                logging.error(f"Google Vision Client başlatılamadı: {e}")
                self.ocr_enabled = False
        else:
            logging.info("Google Vision OCR devre dışı bırakıldı veya credentials eksik.")

        # HuggingFace çeviri modeli – yalnızca translation modeli sağlanmışsa
        if self.translation_enabled and self.translation_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.translation_model)
                self.translation_model_instance = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model)
                logging.info("HuggingFace çeviri modeli yüklendi.")
            except Exception as e:
                logging.error(f"HuggingFace çeviri modeli yüklenemedi: {e}")
                self.translation_enabled = False
        else:
            logging.info("Çeviri modeli devre dışı bırakıldı veya belirtilmemiş.")

    def process_image(self, image_path):
        results = {}
        # OCR işlemi
        if self.ocr_enabled:
            start_time = time.time()
            ocr_text = self.run_ocr(image_path)
            elapsed = time.time() - start_time
            cost = self.cost_evaluator.evaluate_cost("ocr", ocr_text)
            results["ocr"] = {
                "extracted_text": ocr_text,
                "time_spent_sec": elapsed,
                "cost": cost,
                "text_length": len(ocr_text)
            }
            logging.info(f"OCR tamamlandı: {elapsed:.2f} s, Maliyet = {cost}, Karakter = {len(ocr_text)}")
        else:
            logging.info("OCR işlemi atlanıyor (devre dışı).")

        # Çeviri işlemi
        if self.translation_enabled and results.get("ocr", {}).get("extracted_text", "").strip():
            start_time = time.time()
            translation = self.translate_text(results["ocr"]["extracted_text"])
            elapsed = time.time() - start_time
            cost = self.cost_evaluator.evaluate_cost("translation", translation)
            results["translation"] = {
                "translated_text": translation,
                "time_spent_sec": elapsed,
                "cost": cost,
                "text_length": len(translation)
            }
            logging.info(f"Çeviri tamamlandı: {elapsed:.2f} s, Maliyet = {cost}, Karakter = {len(translation)}")
        else:
            logging.info("Çeviri işlemi atlanıyor (devre dışı veya OCR sonucu boş).")

        return results

    def run_ocr(self, image_path):
        try:
            with open(image_path, "rb") as f:
                content = f.read()
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(
                image=image,
                image_context={"language_hints": self.language_hints}
            )
            texts = [ann.description for ann in response.text_annotations]
            return "\n".join(texts[1:]) if texts else ""
        except Exception as e:
            logging.error(f"OCR sırasında hata: {e}")
            return ""

    def translate_text(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self.translation_model_instance.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Çeviri sırasında hata: {e}")
            return ""


########################################
# 6. ANA AKIŞ (MAIN)
########################################

def main():
    # Konfigürasyonu yükle ve loglama ayarla.
    config_manager = ConfigManager("config.json")
    config = config_manager.get()
    setup_logging(config.get("general", {}).get("log_folder", "logs"))
    logging.info("Uygulama başladı.")

    # CostEvaluator instance'ını oluştur (manuel maliyet değerlerini config'den okuyacak).
    cost_evaluator = CostEvaluator(config)

    # Pipeline'ları başlat
    prompt_pipeline = PromptPipeline(config, cost_evaluator)
    ocr_translation_pipeline = OCRTranslationPipeline(config, cost_evaluator)

    img_folder = config.get("general", {}).get("img_folder", "IMG")
    allowed_exts = config.get("general", {}).get("allowed_extensions", [".jpg", ".jpeg", ".png"])

    all_results = {}

    for filename in os.listdir(img_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_exts:
            logging.info(f"{filename} uygun uzantıda değil, atlanıyor.")
            continue

        image_path = os.path.join(img_folder, filename)
        logging.info(f"İşleniyor: {image_path}")
        print(f"İşleniyor: {image_path}")

        # Tek bir prompt tüm modeller için kullanılacak.
        user_prompt = None  # Eğer kullanıcıdan prompt alınmazsa, standart prompt kullanılır.

        # Pipeline 1: Prompt API'leri (Gemini, OpenAI, Anthropic, HuggingFace)
        prompt_results = prompt_pipeline.process_image(image_path, user_prompt)

        # Pipeline 2: OCR & Çeviri
        ocr_translation_results = ocr_translation_pipeline.process_image(image_path)

        # Görsel bazında sonuçları birleştir
        image_result = {
            "prompt_pipeline": prompt_results,
            "ocr_translation_pipeline": ocr_translation_results
        }
        all_results[filename] = image_result

        # Görsel bazında JSON sonuç dosyası oluştur
        result_file = os.path.join(config.get("general", {}).get("results_folder", "results"),
                                   f"analysis_{os.path.splitext(filename)[0]}.json")
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(image_result, f, indent=2, ensure_ascii=False)
            logging.info(f"Sonuç dosyası kaydedildi: {result_file}")
        except Exception as e:
            logging.error(f"Sonuç dosyası kaydedilemedi: {e}")

    # Tüm sonuçları tek bir dosyada da saklama
    all_results_file = os.path.join(config.get("general", {}).get("results_folder", "results"), "all_results.json")
    try:
        with open(all_results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Tüm sonuçlar {all_results_file} dosyasına kaydedildi.")
    except Exception as e:
        logging.error(f"Tüm sonuçlar kaydedilemedi: {e}")

    # Terminale özet bilgileri yazdırma
    print("\n--- Özet Bilgiler ---")
    for img, res in all_results.items():
        print(f"\nResim: {img}")
        for pipe_name, metrics in res.items():
            print(f"  Pipeline: {pipe_name}")
            for model, values in metrics.items():
                time_spent = values.get('time_spent_sec', 'N/A')
                cost = values.get('cost', 'N/A')
                token_info = values.get('token_size', values.get('text_length', 'N/A'))
                print(f"    {model} -> Süre: {time_spent:.2f} s, Maliyet: {cost}, Token/Char Sayısı: {token_info}")

    logging.info("İşlem tamamlandı.")
    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()