import asyncio
from datetime import datetime
from typing import Literal, List
import numpy as np
import struct

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
        '-cluster_time_limit', '1000',  # Лимит времени кластера в миллисекундах (1 секунда)
        '-cluster_size_limit', '32768',  # Лимит размера кластера в байтах (32KB)
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


def split_mp4_into_chunks(encoded_data: bytes, max_chunk_size: int = 65536) -> List[bytes]:
    """
    Разбивает фрагментированный MP4 на чанки по границам moof боксов.
    Первый чанк содержит moov + первый moof, остальные - последующие moof/mdat пары.
    """
    chunks = []
    data = bytearray(encoded_data)
    pos = 0
    first_moof_pos = None

    # Ищем первый moof бокс
    while pos < len(data) - 8:
        # Читаем размер бокса (4 байта, big-endian)
        if pos + 8 > len(data):
            break
        box_size = struct.unpack('>I', data[pos:pos+4])[0]
        box_type = data[pos+4:pos+8]

        if box_type == b'moof':
            first_moof_pos = pos
            break

        if box_size == 0 or box_size > len(data) - pos:
            pos += 1
            continue

        pos += box_size

    if first_moof_pos is None:
        # Если moof не найден, возвращаем весь файл как один чанк
        return [encoded_data]

    # Первый чанк: всё до первого moof включительно
    # Находим конец первого moof (ищем следующий moof или конец файла)
    pos = first_moof_pos
    first_chunk_end = first_moof_pos

    while pos < len(data) - 8:
        box_size = struct.unpack('>I', data[pos:pos+4])[0]
        box_type = data[pos+4:pos+8]

        if box_size == 0 or box_size > len(data) - pos:
            break

        first_chunk_end = pos + box_size

        # Если нашли следующий moof, останавливаемся
        if box_type == b'moof' and pos > first_moof_pos:
            break

        pos += box_size

    # Первый чанк: moov + первый moof
    first_chunk = bytes(data[:first_chunk_end])
    chunks.append(first_chunk)

    # Остальные чанки: последующие moof/mdat пары
    pos = first_chunk_end
    while pos < len(data) - 8:
        chunk_start = pos

        # Находим следующий moof
        while pos < len(data) - 8:
            box_size = struct.unpack('>I', data[pos:pos+4])[0]
            box_type = data[pos+4:pos+8]

            if box_size == 0 or box_size > len(data) - pos:
                # Дошли до конца, добавляем остаток
                if pos < len(data):
                    chunks.append(bytes(data[chunk_start:]))
                break

            pos += box_size

            # Если нашли moof, начинаем новый чанк
            if box_type == b'moof':
                # Завершаем предыдущий чанк
                if chunk_start < pos - box_size:
                    chunks.append(bytes(data[chunk_start:pos - box_size]))
                chunk_start = pos - box_size
            elif pos >= len(data):
                # Конец файла
                if chunk_start < len(data):
                    chunks.append(bytes(data[chunk_start:]))
                break
        else:
            # Добавляем последний чанк
            if chunk_start < len(data):
                chunks.append(bytes(data[chunk_start:]))
            break

    return chunks if chunks else [encoded_data]


def split_webm_into_chunks(encoded_data: bytes, max_chunk_size: int = 65536) -> List[bytes]:
    """
    Разбивает WebM на чанки по границам кластеров.
    Первый чанк содержит заголовок + первый кластер, остальные - последующие кластеры.
    Если кластер один большой, разбивает его на части фиксированного размера.
    """
    chunks = []
    data = bytearray(encoded_data)
    pos = 0
    first_cluster_pos = None

    # Ищем первый кластер (EBML ID: 0x1F43B675)
    cluster_marker = bytes([0x1F, 0x43, 0xB6, 0x75])
    while pos < len(data) - 4:
        if data[pos:pos+4] == cluster_marker:
            first_cluster_pos = pos
            break
        pos += 1

    if first_cluster_pos is None:
        # Если кластер не найден, возвращаем весь файл как один чанк
        return [encoded_data]

    # Находим все кластеры
    cluster_positions = [first_cluster_pos]
    search_pos = first_cluster_pos + 4
    while search_pos < len(data) - 4:
        if data[search_pos:search_pos+4] == cluster_marker:
            cluster_positions.append(search_pos)
        search_pos += 1

    # Если только один кластер, разбиваем его на части
    if len(cluster_positions) == 1:
        # Первый чанк: заголовок + начало первого кластера (до max_chunk_size)
        header_end = first_cluster_pos
        first_chunk_size = min(max_chunk_size, len(data) - header_end)
        first_chunk = bytes(data[:header_end + first_chunk_size])
        chunks.append(first_chunk)

        # Остальные чанки: продолжение кластера
        pos = header_end + first_chunk_size
        while pos < len(data):
            chunk_size = min(max_chunk_size, len(data) - pos)
            chunks.append(bytes(data[pos:pos + chunk_size]))
            pos += chunk_size

        return chunks

    # Если несколько кластеров, разбиваем по границам кластеров
    # Первый чанк: заголовок + первый кластер
    if len(cluster_positions) > 1:
        first_chunk_end = cluster_positions[1]
    else:
        first_chunk_end = len(data)

    first_chunk = bytes(data[:first_chunk_end])
    chunks.append(first_chunk)

    # Остальные чанки: последующие кластеры
    for i in range(1, len(cluster_positions)):
        cluster_start = cluster_positions[i]
        if i + 1 < len(cluster_positions):
            cluster_end = cluster_positions[i + 1]
        else:
            cluster_end = len(data)

        cluster_data = bytes(data[cluster_start:cluster_end])

        # Если кластер большой, разбиваем его на части
        if len(cluster_data) > max_chunk_size:
            chunk_start = 0
            while chunk_start < len(cluster_data):
                chunk_size = min(max_chunk_size, len(cluster_data) - chunk_start)
                chunks.append(cluster_data[chunk_start:chunk_start + chunk_size])
                chunk_start += chunk_size
        else:
            chunks.append(cluster_data)

    return chunks if chunks else [encoded_data]


async def convert_audio_to_chunks(
    audio_np: np.ndarray,
    format: Literal['mp4_opus', 'webm_opus', 'mp3'] = 'webm_opus',
    max_chunk_size: int = 65536
) -> List[bytes]:
    """
    Конвертирует numpy массив аудио в формат и разбивает на чанки по границам фрагментов.
    Возвращает список чанков, готовых для последовательной отправки через MSE.
    """
    # Кодируем всё аудио целиком
    encoded_data = await convert_audio_to_mp4_opus(audio_np, format)

    # Разбиваем на чанки по границам фрагментов
    if format == 'mp4_opus':
        chunks = split_mp4_into_chunks(encoded_data, max_chunk_size)
    elif format == 'webm_opus':
        chunks = split_webm_into_chunks(encoded_data, max_chunk_size)
    else:
        # Для MP3 просто разбиваем по размеру
        chunks = []
        for i in range(0, len(encoded_data), max_chunk_size):
            chunks.append(encoded_data[i:i + max_chunk_size])

    print(f"Аудио закодировано: {len(encoded_data)} байт, разбито на {len(chunks)} чанков")
    if chunks:
        print(f"Размеры чанков: {[len(c) for c in chunks[:5]]}{'...' if len(chunks) > 5 else ''}")

    return chunks

