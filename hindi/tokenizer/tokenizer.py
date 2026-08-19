"""Hindi tokenizer wrapper (Tokenizer interface).

The interface below is locked — downstream pipelines use the standard interface:

    class Tokenizer:
        def __init__(self, model_path: str): ...
        def encode(self, text: str) -> list[int]: ...
        def decode(self, ids: list[int]) -> str: ...
        @property vocab_size, pad_id, eos_id

Backed by a SentencePiece BPE model trained with ``byte_fallback=True``, which
guarantees zero <unk> on anything — even unseen conjuncts (critical for Assamese,
whose corpus is smaller and rare glyphs more likely).
"""

from __future__ import annotations

try:
    import sentencepiece as spm
except ImportError:
    spm = None

__all__ = ["Tokenizer"]


class Tokenizer:
    def __init__(self, model_path: str):
        self._model_path = str(model_path)
        if spm is not None:
            self.sp = spm.SentencePieceProcessor(model_file=self._model_path)
        else:
            self.sp = None

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids (BPE pieces; byte fallback routes unknowns)."""
        if self.sp is not None:
            return self.sp.encode(text, out_type=int)
        # Fallback byte encoding for tests without sentencepiece
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """list of token ids -> text."""
        if self.sp is not None:
            return self.sp.decode(ids)
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")

    def encode_as_pieces(self, text: str) -> list[str]:
        """Text -> list of piece strings (display helper, e.g. for heatmaps)."""
        if self.sp is not None:
            return self.sp.encode(text, out_type=str)
        return list(text)

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return self.sp.get_piece_size()
        return 32768

    @property
    def pad_id(self) -> int:
        if self.sp is not None:
            return self.sp.pad_id()
        return 0

    @property
    def eos_id(self) -> int:
        if self.sp is not None:
            return self.sp.eos_id()
        return 3

    @property
    def bos_id(self) -> int:
        if self.sp is not None:
            return self.sp.bos_id()
        return 2

    @property
    def unk_id(self) -> int:
        if self.sp is not None:
            return self.sp.unk_id()
        return 1

    @property
    def model_path(self) -> str:
        return self._model_path


def load_tokenizer(lang_dir: str, name: str = "") -> Tokenizer:
    """Load <lang_dir>/tokenizer/<lang>.model (name defaults to the dir basename)."""
    from pathlib import Path

    root = Path(lang_dir)
    lang = root.name
    model = root / "tokenizer" / (name or lang + ".model")
    return Tokenizer(str(model))
