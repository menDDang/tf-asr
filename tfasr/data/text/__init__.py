from __future__ import absolute_import


import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = text.strip("\n")
    return text


from .char_tokenizer import CharTokenizer