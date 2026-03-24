import logging
import os
from typing import Dict
import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

from app.core.models import ChunkTiming, GeneratorChunk

logger = logging.getLogger(__name__)


def load_xvector_prompt(path: str, device: str = "cuda") -> dict:
    """Загружает precomputed speaker embedding из .pt файла."""
    spk_emb = torch.load(path, weights_only=True).to(device)
    return {
        "ref_spk_embedding": [spk_emb],
    }


class QwenTTS:
    def __init__(self):
        model_path = os.getenv("qwen_tts_model_path")
        if model_path is None:
            raise ValueError("qwen_tts_model_path is not set")
        self.model = FasterQwen3TTS.from_pretrained(model_path, device="cuda:0")
        self.audio_prompts = self.get_audio_prompts()
        self._warmup()

    def _warmup(self):
        audio_prompt = list(self.audio_prompts.values())[0]
        self.model.generate_voice_clone(
            text="warmup",
            language="auto",
            voice_clone_prompt=audio_prompt,
        )

    def get_audio_prompts(self) -> Dict[str, dict]:
        prompts_path = os.getenv("audio_prompts_path")
        if prompts_path is None:
            raise ValueError("audio_prompts_path is not set")

        audio_prompts = {}
        for speaker_name_dir in os.listdir(prompts_path):
            speaker_file_path = os.path.join(
                prompts_path, speaker_name_dir, f"{speaker_name_dir}.pt"
            )
            if not os.path.exists(speaker_file_path):
                continue
            audio_prompts[speaker_name_dir] = load_xvector_prompt(
                speaker_file_path, device="cuda:0"
            )
        return audio_prompts

    def _max_new_tokens_from_prepare(self, text: str, voice_clone_prompt: dict) -> int:
        """
        Оценивает max_new_tokens по длине trailing_text_hiddens после того же
        _prepare_generation, что и generate_voice_clone (faster_qwen3_tts).

        В decode каждый шаг увеличивает gen_step; пока gen_step < T — подмешивается
        текст; дальше идёт до EOS. Нужен запас шагов на акустику после текста.

        Переменные окружения (опционально):
        - TTS_USE_PREPARE_MAX_TOKENS: "1" (по умолчанию) — считать; "0" — только cap.
        - TTS_MAX_NEW_TOKENS_CAP: верхняя граница (по умолчанию 720 ≈ 60 с при 12 Hz).
        - TTS_MAX_NEW_TOKENS_MIN: нижняя граница (24).
        - TTS_MAX_NEW_TOKENS_EXTRA_BASE: шаги после текста (96).
        - TTS_MAX_NEW_TOKENS_TRAILING_SCALE: доля от trailing (0.35).
        """
        cap = int(os.getenv("TTS_MAX_NEW_TOKENS_CAP", "720"))
        if os.getenv("TTS_USE_PREPARE_MAX_TOKENS", "1").strip().lower() in (
            "0",
            "false",
            "no",
        ):
            return cap

        min_v = int(os.getenv("TTS_MAX_NEW_TOKENS_MIN", "24"))
        extra_base = int(os.getenv("TTS_MAX_NEW_TOKENS_EXTRA_BASE", "96"))
        scale = float(os.getenv("TTS_MAX_NEW_TOKENS_TRAILING_SCALE", "0.35"))

        with torch.inference_mode():
            _, _, _, _, _, tth, _, _ = self.model._prepare_generation(
                text=text,
                language="auto",
                voice_clone_prompt=voice_clone_prompt,
                non_streaming_mode=False,
                append_silence=True,
                xvec_only=False,
            )

        trailing = int(tth.shape[1])
        extra_scaled = int(trailing * scale)
        raw = trailing + extra_base + extra_scaled
        n = min(cap, max(min_v, raw))
        logger.debug(
            "max_new_tokens: trailing_text_hiddens=%s raw=%s -> %s (cap=%s)",
            trailing,
            raw,
            n,
            cap,
        )
        return n

    def generate(self, text: str, speaker_name: str):
        vcp = self.audio_prompts[speaker_name]
        if vcp is None:
            raise ValueError(f"Voice clone prompt not found for {speaker_name}")
        max_new_tokens = self._max_new_tokens_from_prepare(text, vcp)
        audio_list, sr = self.model.generate_voice_clone(
            text=text,
            language="auto",
            voice_clone_prompt=vcp,
            max_new_tokens=max_new_tokens,
        )
        raw = audio_list[0]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        arr = np.asarray(raw, dtype=np.float32)
        return arr.copy(), sr

    def generate_streaming(self, text: str, speaker_name: str):
        vcp = self.audio_prompts[speaker_name]
        if vcp is None:
            raise ValueError(f"Voice clone prompt not found for {speaker_name}")
        max_new_tokens = self._max_new_tokens_from_prepare(text, vcp)
        for audio_chunk, sr, timing in self.model.generate_voice_clone_streaming(
            text=text,
            language="auto",
            voice_clone_prompt=vcp,
            chunk_size=8,
            max_new_tokens=max_new_tokens,
        ):
            yield GeneratorChunk(
                audio_chunk=audio_chunk,
                sr=sr,
                timing=ChunkTiming(chunk_index=timing["chunk_index"], is_final=timing["is_final"]),
            )
