"""
MambaFlow-TTS Text & Alignment Preprocessing Module.
"""

try:
    from .text import text_to_sequence, sequence_to_text, cleaned_text_to_sequence
    from .monotonic_align import maximum_path
except (ImportError, ValueError):
    from preprocessing.text import text_to_sequence, sequence_to_text, cleaned_text_to_sequence
    from preprocessing.monotonic_align import maximum_path

__all__ = [
    "text_to_sequence",
    "sequence_to_text",
    "cleaned_text_to_sequence",
    "maximum_path",
]
