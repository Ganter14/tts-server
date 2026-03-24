"""Монотонные интервалы для логов (time.perf_counter)."""

import time


def perf_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000
