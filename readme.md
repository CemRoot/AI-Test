**Diller / Languages:**
- [Türkçe](#görsel-analiz-ve-ocr-çözümleme-platformu)
- [English](#visual-analysis-and-ocr-processing-platform)

# Visual Analysis and OCR Processing Platform

An integrated platform for visual analysis using multiple AI providers (Google Gemini, OpenAI, Anthropic) and OCR/translation services.

## 🌟 Features
- **Multi-Model Support**: Gemini, OpenAI, Anthropic, HuggingFace
- **OCR Integration**: Text extraction with Google Vision API
- **Translation**: HF Models and Google Translate integration
- **Cost Tracking**: Real-time cost calculation per operation
- **Log Management**: Detailed logging and reporting system

## ⚙️ Setup
1. **Requirements**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **API Keys**:
Create `.env` file:
```ini
GOOGLE_APPLICATION_CREDENTIALS="keys/google_service_account.json"
GOOGLE_TRANSLATE_API_KEY="your_google_translate_key"
```

3. **Config File**:
Copy config example:
```bash
cp config.example.json config.json
```

## 🔧 Configuration
Main configuration file (`config.json`) structure:

```json
{
  "models": {
    "gemini": {
      "enabled": true,
      "api_key": "${GEMINI_API_KEY}",
      "prompt_template": "Please analyze the image in detail:"
    },
    "openai": {
      "enabled": false,
      "api_key": "${OPENAI_API_KEY}",
      "endpoint": "https://api.openai.com/v1/chat/completions",
      "prompt_template": "Analyze the image according to the following criteria:"
    },
    "anthropic": {
      "enabled": true,
      "api_key": "",
      "endpoint": "https://api.anthropic.com/v1/messages",
      "prompt_template": "Please examine and analyze the image:"
    },
    "huggingface": {
      "enabled": true,
      "models": [
        {
          "name": "BLIP-Base",
          "model": "Salesforce/blip-image-captioning-base"
        }
      ]
    }
  },
  "ocr": {
    "google": {
      "enabled": true,
      "credentials": "path/to/google_service_account.json",
      "language_hints": ["en", "tr"]
    }
  },
  "translation": {
    "enabled": true,
    "hf_model": "Helsinki-NLP/opus-mt-en-tr"
  },
  "google_translate": {
    "enabled": false,
    "endpoint": "https://translation.googleapis.com/language/translate/v2"
  },
  "general": {
    "img_folder": "IMG",
    "results_folder": "results",
    "log_folder": "logs",
    "allowed_extensions": [".jpg", ".jpeg", ".png"]
  },
  "cost": {
    "gemini": {
      "variant": "gemini-1.5-flash",
      "default_variant": "gemini-1.5-flash",
      "variants": {
        "gemini-1.5-pro": {
          "token_threshold": 128000,
          "input_rate_low": 1.25,
          "input_rate_high": 2.50,
          "output_rate_low": 5.00,
          "output_rate_high": 10.00
        },
        "gemini-1.5-flash": {
          "token_threshold": 128000,
          "input_rate_low": 0.075,
          "input_rate_high": 0.15,
          "output_rate_low": 0.30,
          "output_rate_high": 0.60
        }
      }
    },
    "openai": {
      "divisor": 1000000,
      "variant": "gpt-4o",
      "rates": {
        "gpt-4o": {
          "input_rate": 5.00,
          "output_rate": 15.00
        }
      }
    },
    "anthropic": {
      "divisor": 1000000,
      "model_variant": "claude-3-haiku-20240307",
      "rates": {
        "claude-3-haiku-20240307": {
          "input_rate": 0.25,
          "output_rate": 1.25
        }
      }
    },
    "huggingface": 0.001,
    "ocr": {
      "free_units": 1000,
      "rate_low": 1.50,
      "rate_high": 0.60,
      "tier_threshold": 1000000,
      "units_per_page": 5
    },
    "translation": 0.0003
  }
}
```

## 🚀 Usage
```bash
python main.py
```

### Image Requirements
- Place images in `IMG/` folder
- Supported formats: JPG, PNG, JPEG

## 🔑 API Keys
1. **Google Cloud**:
   - Create service account via [Google Cloud Console](https://console.cloud.google.com/)
   - Enable `Vision API` and `Generative Language API`

2. **Anthropic**:
   - Get API key from [Anthropic Console](https://console.anthropic.com/)

3. **OpenAI** (Optional):
   - Create key via [OpenAI Platform](https://platform.openai.com/api-keys)

## 📊 Cost Management
Cost parameters in `config.json`:
```json
"cost": {
  "gemini": {
    "variants": {
      "gemini-1.5-flash": {
        "input_rate_low": 0.075,
        "output_rate_low": 0.30
      }
    }
  },
  "anthropic": {
    "rates": {
      "claude-3-haiku": {
        "input_rate": 0.25,
        "output_rate": 1.25
      }
    }
  }
}
```

## 📂 File Structure
```
├── IMG/                   # Input images
├── results/               # Analysis results
├── logs/                  # System logs
├── config.json            # Main configuration
└── ocr_usage.json         # OCR usage tracking
```

## 🛠️ Troubleshooting
- **OCR Errors**: Verify Google service account path
- **Model Loading Issues**: Run `pip install --upgrade transformers torch`
- **Permission Issues**: Ensure `.env` file is in correct location

## 🤝 Contributing
To contribute:
1. Fork the repo
2. Create new branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add some feature'`
4. Push: `git push origin feature/new-feature`
5. Create Pull Request

## 🔓 Open Source
This project is developed as open source. You can freely use and modify it.


---------------------

---------------------
# Görsel Analiz ve OCR Çözümleme Platformu

Çoklu AI sağlayıcıları (Google Gemini, OpenAI, Anthropic) ve OCR/Çeviri servisleri kullanarak görsel analiz yapan entegre bir platform.

## 🌟 Özellikler
- **Çoklu Model Destek**: Gemini, OpenAI, Anthropic ve HuggingFace
- **OCR Entegrasyon**: Google Vision API ile metin çıkarımı
- **Çeviri**: HF Modelleri ve Google Translate entegrasyonu
- **Maliyet Takip**: Her işlem için gerçek zamanlı maliyet hesaplama
- **Log Yönetimi**: Detaylı loglama ve raporlama sistemi

## ⚙️ Kurulum
1. **Gereksinimler**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **API Anahtarları**:
`.env` dosyası oluşturun:
```ini
GOOGLE_APPLICATION_CREDENTIALS="keys/google_service_account.json"
GOOGLE_TRANSLATE_API_KEY="your_google_translate_key"
```

3. **Config Dosyası**:
`config.json` örneğini kopyalayın:
```bash
cp config.example.json config.json
```

## 🔧 Yapılandırma
Ana yapılandırma dosyası (`config.json`) yapısı:

```json:config.json
{
  "models": {
    "gemini": {"enabled": true},
    "openai": {"enabled": false},
    "anthropic": {"enabled": true},
    "huggingface": {
      "models": [{"name": "BLIP-Base", "model": "Salesforce/blip-image-captioning-base"}]
    }
  },
  "ocr": {
    "google": {
      "enabled": true,
      "credentials": "path/to/google_key.json",
      "language_hints": ["tr", "en"]
    }
  },
  "translation": {
    "enabled": true,
    "hf_model": "Helsinki-NLP/opus-mt-tr-en"
  }
}
```

## 🚀 Kullanım
```bash
python main.py
```

### Görsel Gereksinimleri
- Görseller `IMG/` klasörüne yerleştirilmeli
- Desteklenen formatlar: JPG, PNG, JPEG

## 🔑 API Anahtarları
1. **Google Cloud**:
   - [Google Cloud Console](https://console.cloud.google.com/) üzerinden servis hesabı oluşturun
   - `Vision API` ve `Generative Language API` etkinleştirin

2. **Anthropic**:
   - [Anthropic Console](https://console.anthropic.com/) üzerinden API anahtarı alın

3. **OpenAI** (Opsiyonel):
   - [OpenAI Platform](https://platform.openai.com/api-keys) üzerinden anahtar oluşturun

## 📊 Maliyet Yönetimi
Maliyet parametreleri `config.json` içinde ayarlanır:
```json
"cost": {
  "gemini": {
    "variants": {
      "gemini-1.5-flash": {
        "input_rate_low": 0.075,
        "output_rate_low": 0.30
      }
    }
  },
  "anthropic": {
    "rates": {
      "claude-3-haiku": {
        "input_rate": 0.25,
        "output_rate": 1.25
      }
    }
  }
}
```

## 📂 Dosya Yapısı
```
├── IMG/                   # Giriş görselleri
├── results/               # Analiz sonuçları
├── logs/                  # Sistem logları
├── config.json            # Ana yapılandırma
└── ocr_usage.json         # OCR kullanım takibi
```

## 🛠️ Sorun Giderme
- **OCR Hatası**: Google servis hesabı dosya yolunu kontrol edin
- **Model Yükleme Hatası**: `pip install --upgrade transformers torch`
- **Yetki Sorunları**: `.env` dosyasının doğru konumda olduğundan emin olun

## 🤝 Katkı
Katkıda bulunmak için:
1. Repoyu fork'layın
2. Yeni branch oluşturun: `git checkout -b feature/new-feature`
3. Değişiklikleri commit edin: `git commit -m 'Add some feature'`
4. Push işlemi: `git push origin feature/new-feature`
5. Pull Request oluşturun

## 📜 Lisans
MIT Lisansı - Detaylar için [LICENSE](LICENSE) dosyasına bakınız
