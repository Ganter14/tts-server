import asyncio
import datetime
import os
from typing import Literal, List
import numpy as np
import ffmpeg

async def convert_audio_to_opus(audio_np: np.ndarray) -> bytes:
    process = None
    try:
        process = (
            ffmpeg
            .input(
                'pipe:',
                format='f32le',
                ar=24000,
                ac=1
            )
            .output(
                'pipe:1',
                format='mp4',
                ar=48000,
                ac=1,
                acodec='libopus',
                audio_bitrate='64k',
                frame_duration=20,
                application='voip',
                packet_loss=0,
                movflags='frag_keyframe+empty_moov+default_base_moof',
                cluster_time_limit='1000',
                cluster_size_limit='32768'
            )
            .run_async(pipe_stdout=True, pipe_stderr=True, pipe_stdin=True)
        )
        print(f"process: {process}")
        audio_bytes = audio_np.tobytes()
        stdout, stderr = await asyncio.to_thread(process.communicate, input=audio_bytes)

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg конвертация не удалась (код {process.returncode}): {error_msg}")
        return stdout

    except FileNotFoundError:
        raise RuntimeError("ffmpeg не найден. Убедитесь что ffmpeg установлен и доступен в PATH")
    except Exception as e:
        raise RuntimeError(f"Ошибка при конвертации аудио в MP4/Opus: {e}")
    finally:
        if process:
            process.terminate()


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
    encoded_data = await convert_audio_to_opus(audio_np)

    if os.getenv("is_save_audio") == "True":
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now().strftime("%H-%M-%S")
        audio_dir = f"generated_audio/{current_date}/ffmpeg"
        os.makedirs(audio_dir, exist_ok=True)
        with open(f"{audio_dir}/{current_time}.webm", "wb") as f:
            f.write(encoded_data)

    chunks = split_webm_into_chunks(encoded_data, max_chunk_size)

    print(f"Аудио закодировано: {len(encoded_data)} байт, разбито на {len(chunks)} чанков")
    if chunks:
        print(f"Размеры чанков: {[len(c) for c in chunks[:5]]}{'...' if len(chunks) > 5 else ''}")

    return chunks

