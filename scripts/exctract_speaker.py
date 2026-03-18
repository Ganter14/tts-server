# init dotenv
import os
from dotenv import load_dotenv
import torch
from qwen_tts import Qwen3TTSModel

load_dotenv()

model_path = os.getenv("qwen_tts_model_path")
if model_path is None:
    raise ValueError("qwen_tts_model_path is not set")

model = Qwen3TTSModel.from_pretrained(model_path)

audio_prompts_path = os.getenv("audio_prompts_path")
if audio_prompts_path is None:
    raise ValueError("audio_prompts_path is not set")

for speaker_name in os.listdir(audio_prompts_path):
    audio_prompt_path = os.path.join(audio_prompts_path, speaker_name, f"{speaker_name}.wav")
    if not os.path.exists(audio_prompt_path):
        print(f"Audio prompt not found for {speaker_name}")
        continue
    ref_text_path = os.path.join(audio_prompts_path, speaker_name, f"{speaker_name}.txt")
    if not os.path.exists(ref_text_path):
        print(f"Ref text not found for {speaker_name}")
        continue
    ref_text = open(ref_text_path, "r").read()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=audio_prompt_path,
        ref_text=ref_text,
    )
    spk_emb = prompt_items[0].ref_spk_embedding.cpu()
    torch.save(spk_emb, os.path.join(audio_prompts_path, speaker_name, f"{speaker_name}.pt"))
    print(f"Speaker embedding saved to {os.path.join(audio_prompts_path, speaker_name, f"{speaker_name}.pt")}")
