import unicodedata


def normalize_tts_text(text: str) -> str:
    """
    NFKC, удаление невидимых/форматных символов (Cf, Mn — в т.ч. U+034F CGJ),
    схлопывание пробелов.
    """
    s = unicodedata.normalize("NFKC", text)
    s = "".join(
        ch for ch in s if unicodedata.category(ch) not in ("Cf", "Mn", "Me")
    )
    return " ".join(s.split()).strip()
