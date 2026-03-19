import os
from typing import Dict

import torch
from faster_qwen3_tts import FasterQwen3TTS


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
      speaker_file_path = os.path.join(prompts_path, speaker_name_dir, f"{speaker_name_dir}.pt")
      if not os.path.exists(speaker_file_path):
        continue
      audio_prompts[speaker_name_dir] = load_xvector_prompt(speaker_file_path, device="cuda:0")
    return audio_prompts

  def generate(self, text: str, speaker_name: str):
    vcp = self.audio_prompts[speaker_name]
    if vcp is None:
      raise ValueError(f"Voice clone prompt not found for {speaker_name}")
    audio_list, sr = self.model.generate_voice_clone(
      text=text,
      language="auto",
      voice_clone_prompt=vcp,
    )
    return audio_list[0], sr

  def generate_streaming(self, text: str, speaker_name: str):
    vcp = self.audio_prompts[speaker_name]
    if vcp is None:
      raise ValueError(f"Voice clone prompt not found for {speaker_name}")
    for audio_chunk, sr, timing in self.model.generate_voice_clone_streaming(
      text=text,
      language="auto",
      voice_clone_prompt=vcp,
      chunk_size=8
    ):
      yield {
        "audio_chunk": audio_chunk,
        "sr": sr,
        "timing": timing,
      }
