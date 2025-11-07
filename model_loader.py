import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def load_model(device="cuda"):
    """Загружает модель TTS"""
    try:
        print("Загружаем модель TTS...")
        model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
        print("Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели на {device}: {e}")
        raise


def load_model_with_fallback():
    """Загружает модель TTS с fallback на CPU если CUDA недоступна"""
    try:
        model = load_model("cuda")
        print("Модель загружена на CUDA")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели на CUDA: {e}")
        # Fallback на CPU если CUDA недоступна
        try:
            model = load_model("cpu")
            print("Модель загружена на CPU")
            return model
        except Exception as e2:
            print(f"Критическая ошибка: {e2}")
            raise

