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
from google.auth.transport.requests import Request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
from dotenv import load_dotenv

# Ortam değişkenlerini .env dosyasından yükle
load_dotenv()

# Global tokenizer for token counting (using GPT-2)
TOKENIZER_FOR_COUNT = AutoTokenizer.from_pretrained("gpt2")


########################################
# 1. GOOGLE OAUTH 2.0 TOKEN ÜRETİMİ
########################################

def get_google_oauth_token():
    """
    GOOGLE_APPLICATION_CREDENTIALS ortam değişkeninde belirtilen JSON anahtar dosyasını
    kullanarak geçerli bir OAuth 2.0 erişim belirteci üretir.
    """
    # .env dosyanızda belirtilen yolu alın (örneğin, "keys/your_service_account.json")
    relative_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not relative_path:
        logging.error("GOOGLE_APPLICATION_CREDENTIALS ortam değişkeni ayarlanmadı!")
        return None

    # main.py dosyasının bulunduğu dizini referans alarak mutlak yol oluşturun
    credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

    if not os.path.exists(credentials_path):
        logging.error(f"Belirtilen Google credential dosyası bulunamadı: {credentials_path}")
        return None

    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language"
    ]
    try:
        credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        logging.error(f"Google OAuth token alınamadı: {e}")
        return None

########################################
# 2. CONFIG YÖNETİCİSİ
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
# 3. LOG YÖNETİMİ
########################################

def setup_logging(log_folder):
    log_filename = os.path.join(log_folder, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(f"Log dosyası: {log_filename}")


########################################
# 4. OCR USAGE KALICILIĞI İÇİN YARDIMCI FONKSİYONLAR
########################################

def get_ocr_usage():
    if os.path.exists("ocr_usage.json"):
        with open("ocr_usage.json", "r") as f:
            usage = json.load(f)
    else:
        usage = {"used_units": 0}
    return usage


def update_ocr_usage(used_units):
    with open("ocr_usage.json", "w") as f:
        json.dump({"used_units": used_units}, f)


########################################
# 5. MALİYET YÖNETİCİSİ (CostEvaluator)
########################################

class CostEvaluator:
    def __init__(self, config):
        self.cost_config = config.get("cost", {})

    def evaluate_cost(self, model_name, response, extra_units=0):
        model = model_name.lower()
        config_entry = self.cost_config.get(model, None)

        # Gemini hesaplaması
        if model == "gemini":
            if config_entry and isinstance(config_entry, dict):
                results = {}
                variants = config_entry.get("variants", {})
                for variant, variant_config in variants.items():
                    if response and "usage" in response:
                        usage = response["usage"]
                        if "input_tokens" in usage and "output_tokens" in usage:
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                        elif "prompt_tokens" in usage and "completion_tokens" in usage:
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)
                        else:
                            input_tokens = 0
                            output_tokens = 0
                        if input_tokens or output_tokens:
                            threshold = variant_config.get("token_threshold", 128000)
                            in_rate = variant_config.get("input_rate_low",
                                                         1.25) if input_tokens <= threshold else variant_config.get(
                                "input_rate_high", 2.50)
                            out_rate = variant_config.get("output_rate_low",
                                                          5.00) if output_tokens <= threshold else variant_config.get(
                                "output_rate_high", 10.00)
                            cost_val = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
                            results[variant] = cost_val
                        else:
                            results[variant] = 0.001
                    else:
                        results[variant] = 0.001
                return results

        # OpenAI hesaplaması
        if model == "openai":
            if config_entry and isinstance(config_entry,
                                           dict) and "rates" in config_entry and response and "usage" in response:
                usage = response["usage"]
                if "prompt_tokens" in usage and "completion_tokens" in usage:
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                else:
                    input_tokens = 0
                    output_tokens = 0
                divisor = config_entry.get("divisor", 1000000)
                variant = config_entry.get("variant", "gpt-4o")
                rates = config_entry["rates"].get(variant, {})
                in_rate = rates.get("input_rate", 5.00)
                out_rate = rates.get("output_rate", 15.00)
                return (input_tokens / float(divisor)) * in_rate + (output_tokens / float(divisor)) * out_rate

        # Anthropic hesaplaması
        if model == "anthropic":
            if config_entry and isinstance(config_entry, dict) and response and "usage" in response:
                usage = response["usage"]
                # API yanıt yapısı Anthropic'e göre düzeltildi
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                model_variant = response.get("model", "claude-3-opus-20240229")
                rates = config_entry.get("rates", {}).get(
                    model_variant,
                    {"input_rate": 15.00, "output_rate": 75.00}
                )

                divisor = config_entry.get("divisor", 1000000)
                in_rate = rates.get("input_rate", 15.00)
                out_rate = rates.get("output_rate", 75.00)

                return (input_tokens / divisor) * in_rate + (output_tokens / divisor) * out_rate

        # HuggingFace ve Translation: sabit maliyet
        if model in ["huggingface", "translation"]:
            if config_entry and not isinstance(config_entry, dict):
                return config_entry

        # OCR hesaplaması (kalıcı kullanım)
        if model == "ocr":
            units_per_page = self.cost_config.get("ocr", {}).get("units_per_page", 5)
            usage = get_ocr_usage()
            cumulative_units = usage.get("used_units", 0)
            new_cumulative = cumulative_units + units_per_page + extra_units
            update_ocr_usage(new_cumulative)
            free_units = self.cost_config.get("ocr", {}).get("free_units", 1000)
            if new_cumulative <= free_units:
                return 0.0
            else:
                billable_units = new_cumulative - free_units
                tier_threshold = self.cost_config.get("ocr", {}).get("tier_threshold", 1000000)
                if billable_units <= (tier_threshold - free_units):
                    return (billable_units / 1000.0) * self.cost_config.get("ocr", {}).get("rate_low", 1.50)
                else:
                    first_part_units = (tier_threshold - free_units)
                    remaining_units = billable_units - first_part_units
                    cost_first = (first_part_units / 1000.0) * self.cost_config.get("ocr", {}).get("rate_low", 1.50)
                    cost_second = (remaining_units / 1000.0) * self.cost_config.get("ocr", {}).get("rate_high", 0.60)
                    return cost_first + cost_second

        return 0.001  # fallback


########################################
# 5.1. VİZÜEL ANALİZ PIPELINE: GEMINI
########################################

class GeminiPipeline:
    """
    Gemini API çağrısını, her varyant için ayrı ayrı gerçekleştirir.
    OAuth 2.0 erişim belirteci kullanılır.
    """

    def __init__(self, config, cost_evaluator):
        self.base_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.cost_evaluator = cost_evaluator
        self.settings = config.get("models", {}).get("gemini", {})
        self.default_prompt = self.settings.get("prompt_template", "Lütfen görseli analiz edin:")

    def process_image(self, image_path, user_prompt=None):
        results = {}
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logging.info(f"[Gemini] Resim {os.path.basename(image_path)} base64'e çevrildi.")
        except Exception as e:
            logging.error(f"[Gemini] Resim okunamadı: {e}")
            return results
        prompt = user_prompt if user_prompt else self.default_prompt
        token = get_google_oauth_token()
        if token is None:
            logging.error("[Gemini] OAuth token alınamadı; işlemi atlanıyor.")
            return results
        headers = {"Authorization": f"Bearer {token}"}
        variants = self.cost_evaluator.cost_config.get("gemini", {}).get("variants", {})
        for variant, variant_config in variants.items():
            endpoint = self.base_endpoint + variant + ":generateContent"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }]
            }
            logging.info(f"[Gemini - {variant}] API çağrısı başlatılıyor.")
            start_time = time.time()
            try:
                response = requests.post(endpoint, headers=headers, json=payload)
                response = response.json()
            except Exception as e:
                logging.error(f"[Gemini - {variant}] API çağrısı sırasında hata: {e}")
                response = {}
            elapsed = time.time() - start_time
            cost_dict = self.cost_evaluator.evaluate_cost("gemini", response)
            variant_cost = cost_dict.get(variant, 0.001) if isinstance(cost_dict, dict) else cost_dict
            results[variant] = {
                "response": response,
                "time_spent_sec": elapsed,
                "cost": variant_cost,
                "token_size": 0
            }
            logging.info(f"[Gemini - {variant}] Süre = {elapsed:.2f} s, Maliyet = {variant_cost}")
        return results


########################################
# 5.2. VİZÜEL ANALİZ PIPELINE: OPENAI
########################################

class OpenAIPipeline:
    def __init__(self, config, cost_evaluator):
        self.settings = config.get("models", {}).get("openai", {})
        self.cost_evaluator = cost_evaluator
        self.default_prompt = self.settings.get("prompt_template", "Lütfen görseli analiz edin:")

    def process_image(self, image_path, user_prompt=None):
        results = {}
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logging.info("[OpenAI] Resim base64'e çevrildi.")
        except Exception as e:
            logging.error(f"[OpenAI] Resim okunamadı: {e}")
            return results
        prompt = user_prompt if user_prompt else self.default_prompt
        api_key = self.settings.get("api_key")
        endpoint = self.settings.get("endpoint")
        if not api_key or not endpoint:
            logging.info("[OpenAI] API anahtarı veya endpoint eksik, atlanıyor.")
            return results
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]
        }
        logging.info("[OpenAI] API çağrısı başlatılıyor.")
        start_time = time.time()
        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response = response.json()
        except Exception as e:
            logging.error(f"[OpenAI] API çağrısı sırasında hata: {e}")
            response = {}
        elapsed = time.time() - start_time
        cost_val = self.cost_evaluator.evaluate_cost("openai", response)
        results["gpt-4o"] = {
            "response": response,
            "time_spent_sec": elapsed,
            "cost": cost_val,
            "token_size": 0
        }
        logging.info(f"[OpenAI] Süre = {elapsed:.2f} s, Maliyet = {cost_val}")
        return results


########################################
# 5.3. VİZÜEL ANALİZ PIPELINE: ANTHROPIC
########################################

class AnthropicPipeline:
    def __init__(self, config, cost_evaluator):
        self.settings = config.get("models", {}).get("anthropic", {})
        self.cost_evaluator = cost_evaluator
        self.default_prompt = self.settings.get("prompt_template", "Lütfen görseli analiz edin:")
        self.endpoint = self.settings.get("endpoint")

        # Model isimlerini güncelle
        self.model_mappings = {
            "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229"
        }

    def process_image(self, image_path, user_prompt=None):
        results = {}
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            logging.info("[Anthropic] Resim base64'e çevrildi.")
        except Exception as e:
            logging.error(f"[Anthropic] Resim okunamadı: {e}")
            return results

        prompt = user_prompt if user_prompt else self.default_prompt
        api_key = self.settings.get("api_key")

        if not api_key or not self.endpoint:
            logging.info("[Anthropic] API anahtarı veya endpoint eksik, atlanıyor.")
            return results

        variants = self.cost_evaluator.cost_config.get("anthropic", {}).get("rates", {})

        for variant, rates in variants.items():
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }

            # Doğru model adını kullan
            model_param = self.model_mappings.get(variant, variant)

            payload = {
                "model": model_param,
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            }

            logging.info(f"[Anthropic - {variant}] API çağrısı başlatılıyor.")
            start_time = time.time()

            try:
                response = requests.post(self.endpoint, headers=headers, json=payload)
                response = response.json()
            except Exception as e:
                logging.error(f"[Anthropic - {variant}] API çağrısı sırasında hata: {e}")
                response = {}

            elapsed = time.time() - start_time
            cost_val = self.cost_evaluator.evaluate_cost("anthropic", response)
            variant_cost = cost_val.get(variant, 0.001) if isinstance(cost_val, dict) else cost_val

            results[variant] = {
                "response": response,
                "time_spent_sec": elapsed,
                "cost": variant_cost,
                "token_size": 0
            }

            logging.info(f"[Anthropic - {variant}] Süre = {elapsed:.2f} s, Maliyet = {variant_cost}")

        return results


########################################
# 5.4. VİZÜEL ANALİZ PIPELINE: HUGGINGFACE
########################################

class HuggingFacePipeline:
    def __init__(self, config, cost_evaluator):
        self.settings = config.get("models", {}).get("huggingface", {})
        self.cost_evaluator = cost_evaluator
        # Qwen2-VL-7B-Instruct modeli devredışı bırakılıyor
        self.models_list = [entry for entry in self.settings.get("models", []) if
                            entry.get("name", "").lower() != "qwen2-vl-7b-instruct"]
        if not self.models_list and self.settings.get("model"):
            self.models_list = [{"name": "default", "model": self.settings.get("model")}]
        self.hf_pipeline = {}
        self.tokenizer_count = TOKENIZER_FOR_COUNT

    def process_image(self, image_path):
        results = {}
        try:
            image = Image.open(image_path)
        except Exception as e:
            logging.error(f"[HuggingFace] Resim okunamadı: {e}")
            return results
        for entry in self.models_list:
            model_name = entry.get("name", entry.get("model", ""))
            model_identifier = entry.get("model", "")
            if not model_identifier:
                logging.info(f"[HuggingFace - {model_name}] Model adı eksik, atlanıyor.")
                continue
            try:
                if model_name not in self.hf_pipeline:
                    try:
                        self.hf_pipeline[model_name] = pipeline("image-to-text", model=model_identifier)
                    except Exception as e:
                        if "MPS backend out of memory" in str(e):
                            logging.error(f"[HuggingFace - {model_name}] MPS hatası, CPU'ya geçiliyor.")
                            self.hf_pipeline[model_name] = pipeline("image-to-text", model=model_identifier, device=-1)
                        else:
                            raise e
                start_time = time.time()
                hf_response = self.hf_pipeline[model_name](image)
                elapsed = time.time() - start_time
                cost_val = self.cost_evaluator.evaluate_cost("huggingface", hf_response)
                generated_text = hf_response[0].get("generated_text", hf_response[0].get("caption", ""))
                token_size = len(self.tokenizer_count.encode(generated_text)) if generated_text else 0
                results[model_name] = {
                    "response": hf_response,
                    "time_spent_sec": elapsed,
                    "cost": cost_val,
                    "token_size": token_size
                }
                logging.info(
                    f"[HuggingFace - {model_name}] Süre = {elapsed:.2f} s, Maliyet = {cost_val}, Token sayısı = {token_size}")
            except Exception as e:
                logging.error(f"[HuggingFace - {model_name}] işlemi sırasında hata: {e}")
                results[model_name] = {"response": {"error": str(e)}}
        return results


########################################
# 6. OCR PIPELINE (Google Vision OCR)
########################################

class OCRPipeline:
    """
    Google Vision OCR kullanarak görselden metin çıkarır.
    """

    def __init__(self, config, cost_evaluator):
        ocr_config = config.get("ocr", {}).get("google", {})
        self.ocr_enabled = ocr_config.get("enabled", False)
        self.google_credentials = ocr_config.get("credentials", None)
        self.language_hints = ocr_config.get("language_hints", ["en"])
        self.cost_evaluator = cost_evaluator
        if self.ocr_enabled and self.google_credentials:
            try:
                credentials = service_account.Credentials.from_service_account_file(self.google_credentials)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                logging.info("[OCR] Google Vision Client başarıyla başlatıldı.")
            except Exception as e:
                logging.error(f"[OCR] Google Vision Client başlatılamadı: {e}")
                self.ocr_enabled = False
        else:
            logging.info("[OCR] Google Vision OCR devre dışı bırakıldı veya credentials eksik.")

    def process_image(self, image_path):
        results = {}
        if self.ocr_enabled:
            start_time = time.time()
            ocr_text = self.run_ocr(image_path)
            elapsed = time.time() - start_time
            cost_val = self.cost_evaluator.evaluate_cost("ocr", ocr_text)
            results["ocr"] = {
                "extracted_text": ocr_text,
                "time_spent_sec": elapsed,
                "cost": cost_val,
                "text_length": len(ocr_text)
            }
            logging.info(f"[OCR] Tamamlandı: {elapsed:.2f} s, Maliyet = {cost_val}, Karakter = {len(ocr_text)}")
        else:
            logging.info("[OCR] İşlem atlanıyor (devre dışı).")
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
            logging.error(f"[OCR] Hata: {e}")
            return ""


########################################
# 7. ÇEVİRİ PIPELINE
########################################

class TranslationPipeline:
    """
    Çeviri iki yöntemle yapılır:
      - HuggingFace tabanlı çeviri (HF)
      - Google Translate API (GT)
    """

    def __init__(self, config, cost_evaluator):
        self.cost_evaluator = cost_evaluator
        translation_config = config.get("translation", {})
        # HF yöntemi
        self.hf_enabled = translation_config.get("enabled", False)
        self.hf_model = translation_config.get("hf_model", None)
        self.hf_tokenizer = None  # Başlangıç değeri
        if self.hf_enabled and self.hf_model:
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                self.hf_model_instance = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model)
                if hasattr(self.hf_model_instance.config, "lang2id"):
                    self.forced_bos_token_id = self.hf_model_instance.config.lang2id.get("eng_Latn", None)
                else:
                    self.forced_bos_token_id = None
                logging.info("[Translation-HF] Model yüklendi.")
            except Exception as e:
                logging.error(f"[Translation-HF] Yüklenemedi: {e}")
                self.hf_enabled = False
        else:
            logging.info("[Translation-HF] Devre dışı veya hf_model belirtilmemiş.")
        # Google Translate yöntemi
        self.gt_enabled = config.get("google_translate", {}).get("enabled", False)
        if self.gt_enabled:
            self.gt_api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
            self.gt_endpoint = config.get("google_translate", {}).get("endpoint",
                                                                      "https://translation.googleapis.com/language/translate/v2")
            logging.info("[Translation-GT] Ayarlar yüklendi.")
        else:
            logging.info("[Translation-GT] Devre dışı.")

    def translate_with_hf(self, text):
        if not self.hf_tokenizer:
            logging.error("[Translation-HF] hf_tokenizer tanımlı değil.")
            return ""
        try:
            inputs = self.hf_tokenizer(text, return_tensors="pt", truncation=True)
            if self.forced_bos_token_id is not None:
                outputs = self.hf_model_instance.generate(**inputs, forced_bos_token_id=self.forced_bos_token_id)
            else:
                outputs = self.hf_model_instance.generate(**inputs)
            return self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"[Translation-HF] Hata: {e}")
            return ""

    def translate_with_gt(self, text):
        try:
            params = {
                "q": text,
                "target": "en",
                "key": self.gt_api_key
            }
            response = requests.post(self.gt_endpoint, params=params)
            result = response.json()
            return result["data"]["translations"][0]["translatedText"]
        except Exception as e:
            logging.error(f"[Translation-GT] Hata: {e}")
            return ""

    def process_text(self, text):
        translations = {}
        if self.hf_enabled:
            start = time.time()
            hf_translation = self.translate_with_hf(text)
            elapsed = time.time() - start
            cost_val = self.cost_evaluator.evaluate_cost("translation", hf_translation)
            translations["huggingface"] = {
                "translated_text": hf_translation,
                "time_spent_sec": elapsed,
                "cost": cost_val,
                "text_length": len(hf_translation)
            }
            logging.info(
                f"[Translation-HF] Süre = {elapsed:.2f} s, Maliyet = {cost_val}, Karakter = {len(hf_translation)}")
        if self.gt_enabled:
            start = time.time()
            gt_translation = self.translate_with_gt(text)
            elapsed = time.time() - start
            cost_val = self.cost_evaluator.evaluate_cost("translation", gt_translation)
            translations["google_translate"] = {
                "translated_text": gt_translation,
                "time_spent_sec": elapsed,
                "cost": cost_val,
                "text_length": len(gt_translation)
            }
            logging.info(
                f"[Translation-GT] Süre = {elapsed:.2f} s, Maliyet = {cost_val}, Karakter = {len(gt_translation)}")
        return translations


########################################
# 8. ANA AKIŞ (MAIN)
########################################

def main():
    config_manager = ConfigManager("config.json")
    config = config_manager.get()
    setup_logging(config.get("general", {}).get("log_folder", "logs"))
    logging.info("Uygulama başladı.")

    cost_evaluator = CostEvaluator(config)
    gemini_pipeline = GeminiPipeline(config, cost_evaluator)
    openai_pipeline = OpenAIPipeline(config, cost_evaluator)
    anthropic_pipeline = AnthropicPipeline(config, cost_evaluator)
    hf_pipeline = HuggingFacePipeline(config, cost_evaluator)
    ocr_pipeline = OCRPipeline(config, cost_evaluator)
    translation_pipeline = TranslationPipeline(config, cost_evaluator)

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
        user_prompt = None  # Eğer belirtilmemişse genel prompt kullanılacak.
        results = {}
        results["Gemini"] = gemini_pipeline.process_image(image_path, user_prompt)
        if config.get("models", {}).get("openai", {}).get("enabled", False):
            results["OpenAI"] = openai_pipeline.process_image(image_path, user_prompt)
        results["Anthropic"] = anthropic_pipeline.process_image(image_path, user_prompt)
        results["HuggingFace"] = hf_pipeline.process_image(image_path)
        results["OCR"] = ocr_pipeline.process_image(image_path)
        ocr_text = results.get("OCR", {}).get("ocr", {}).get("extracted_text", "")
        translations = {}
        if ocr_text.strip():
            translations = translation_pipeline.process_text(ocr_text)
        results["Translation"] = translations

        all_results[filename] = results

        result_file = os.path.join(config.get("general", {}).get("results_folder", "results"),
                                   f"analysis_{os.path.splitext(filename)[0]}.json")
        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"Sonuç dosyası kaydedildi: {result_file}")
        except Exception as e:
            logging.error(f"Sonuç dosyası kaydedilemedi: {e}")

    all_results_file = os.path.join(config.get("general", {}).get("results_folder", "results"), "all_results.json")
    try:
        with open(all_results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Tüm sonuçlar {all_results_file} dosyasına kaydedildi.")
    except Exception as e:
        logging.error(f"Tüm sonuçlar kaydedilemedi: {e}")

    print("\n--- Özet Bilgiler ---")
    for img, res in all_results.items():
        print(f"\nResim: {img}")
        for pipe_name, metrics in res.items():
            print(f"  {pipe_name}:")
            if isinstance(metrics, dict):
                for model, values in metrics.items():
                    time_spent = values.get('time_spent_sec', 'N/A')
                    cost_val = values.get('cost', 'N/A')
                    token_info = values.get('token_size', values.get('text_length', 'N/A'))
                    if isinstance(time_spent, (int, float)):
                        time_str = f"{time_spent:.2f}"
                    else:
                        time_str = time_spent
                    print(f"    {model} -> Süre: {time_str} s, Maliyet: {cost_val}, Token/Char Sayısı: {token_info}")
            else:
                print(f"    {pipe_name}: {metrics}")
    logging.info("İşlem tamamlandı.")
    print("İşlem tamamlandı.")


if __name__ == "__main__":
    main()