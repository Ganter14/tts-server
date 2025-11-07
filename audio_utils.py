import asyncio
from datetime import datetime
from typing import Literal
import numpy as np

mp4_opus_args = [
        'ffmpeg',
        '-f', 'f32le',  # Формат входных данных: 32-bit float little-endian
        '-ar', '24000',  # Входной sample rate (из параметра функции)
        '-ac', '1',  # Моно канал
        '-i', 'pipe:0',  # Читать из stdin
        '-ar', '48000',  # Выходной sample rate (ресемплинг до 48000)
        '-c:a', 'libopus',  # Кодек Opus
        '-b:a', '64k',  # Битрейт для речи
        '-frame_duration', '20',  # Frame duration in milliseconds
        '-application', 'voip',  # Application type
        '-packet_loss', '0',  # Packet loss percentage
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',  # Fragmented MP4 для MSE (без base-data-offset)
        '-f', 'mp4',  # Формат выхода
        'pipe:1'  # Писать в stdout
]

webm_opus_args = [
        'ffmpeg',
        '-f', 'f32le',  # Формат входных данных: 32-bit float little-endian
        '-ar', '24000',  # Входной sample rate (из параметра функции)
        '-ac', '1',  # Моно канал
        '-i', 'pipe:0',  # Читать из stdin
        '-ar', '48000',  # Выходной sample rate (ресемплинг до 48000)
        '-c:a', 'libopus',  # Кодек Opus
        '-b:a', '64k',  # Битрейт для речи
        '-frame_duration', '20',  # Frame duration in milliseconds
        '-application', 'voip',  # Application type
        '-packet_loss', '0',  # Packet loss percentage
        '-f', 'webm',  # Формат выхода (WebM лучше для streaming)
        # Ключевые параметры для фрагментированного WebM:
        '-frag_duration', '1000000',  # Длительность фрагмента в микросекундах (1 секунда)
        '-frag_size', '100000',  # Максимальный размер фрагмента в байтах
        '-reset_timestamps', '1',  # Сброс временных меток для потоковой передачи
        '-cluster_size_limit', '1000000',  # Лимит размера кластера
        '-cluster_time_limit', '1000000',  # Лимит времени кластера
        'pipe:1'  # Писать в stdout
]

mp3_args = [
    'ffmpeg',
    '-f', 'f32le',
    '-ar', '24000',
    '-ac', '1',
    '-i', 'pipe:0',
    '-ar', '48000',
    '-c:a', 'libmp3lame',
    '-b:a', '128k',
    '-f', 'mp3',
    'pipe:1'
]

ffmpeg_args_dict = {
    'mp4_opus': mp4_opus_args,
    'webm_opus': webm_opus_args,
    'mp3': mp3_args,
}

async def convert_audio_to_mp4_opus(audio_np: np.ndarray, format: Literal['mp4_opus', 'webm_opus', 'mp3'] = 'webm_opus') -> bytes:
    """Конвертирует numpy массив аудио в MP4/Opus формат используя ffmpeg"""
    # Убеждаемся что аудио в формате float32
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    # Параметры ffmpeg для конвертации в fragmented MP4 с Opus
    ffmpeg_args = ffmpeg_args_dict[format]

    try:
        # Создаем асинхронный subprocess
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Отправляем аудио данные в stdin и получаем результат из stdout
        stdout, stderr = await process.communicate(input=audio_np.tobytes())

        # Проверяем код возврата
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg конвертация не удалась (код {process.returncode}): {error_msg}")

        # save to file
        now_str = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        extension = 'webm' if format == 'webm_opus' else 'mp4' if format == 'mp4_opus' else 'mp3'
        with open(f"generated_audio/{now_str}.{extension}", "wb") as f:
            f.write(stdout)
        return stdout

    except FileNotFoundError:
        raise RuntimeError("ffmpeg не найден. Убедитесь что ffmpeg установлен и доступен в PATH")
    except Exception as e:
        raise RuntimeError(f"Ошибка при конвертации аудио в MP4/Opus: {e}")

