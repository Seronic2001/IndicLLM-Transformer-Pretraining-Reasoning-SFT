"""Hindi tokenizer wrapper (Tokenizer interface contract).

The interface below is locked — every downstream module imports exactly this:

    class Tokenizer:
        def __init__(self, model_path: str): ...
        def encode(self, text: str) -> list[int]: ...
        def decode(self, ids: list[int]) -> str: ...
        @property vocab_size, pad_id, eos_id

Backed by a SentencePiece BPE model trained with ``byte_fallback=True``, which
guarantees zero <unk> on anything — even unseen conjuncts (critical for Assamese,
whose corpus is smaller and rare glyphs more likely).

Extended with dedicated symbolic logic tokens for chain-of-thought reasoning:
  <COT_START>, <COT_END>, <REL_GT>, <REL_LT>, <REL_EQ>
These are mapped to IDs above the BPE vocabulary ceiling (16384–16388) so that
brackets and relational operators are never fragmented by subword splitting.
"""

from __future__ import annotations

import re

try:
    import sentencepiece as spm
except ImportError:
    spm = None

__all__ = ["Tokenizer"]

# Dedicated symbolic logic tokens — IDs start right after the BPE ceiling (16384)
SPECIAL_LOGIC_TOKENS: dict[str, int] = {
    "<COT_START>": 16384,
    "<COT_END>": 16385,
    "<REL_GT>": 16386,
    "<REL_LT>": 16387,
    "<REL_EQ>": 16388,
    "<REL_DISJOINT>": 16389,
}
_ID_TO_SPECIAL: dict[int, str] = {v: k for k, v in SPECIAL_LOGIC_TOKENS.items()}
_SPECIAL_RE = re.compile("|".join(re.escape(tok) for tok in SPECIAL_LOGIC_TOKENS))
NUM_SPECIAL_LOGIC = len(SPECIAL_LOGIC_TOKENS)


class Tokenizer:
    def __init__(self, model_path: str):
        self._model_path = str(model_path)
        if spm is not None:
            self.sp = spm.SentencePieceProcessor(model_file=self._model_path)
        else:
            self.sp = None

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids (BPE pieces + dedicated logic tokens).

        Special tokens like <COT_START> are intercepted and mapped to their
        atomic IDs *before* the remaining text is passed to SentencePiece, so
        they are never fragmented into subword bytes.
        """
        # Fast path — no special tokens present
        if "<COT_" not in text and "<REL_" not in text:
            if self.sp is not None:
                return self.sp.encode(text, out_type=int)
            return list(text.encode("utf-8"))

        # Split on special tokens, encode each normal segment via SP
        ids: list[int] = []
        parts = _SPECIAL_RE.split(text)
        specials = _SPECIAL_RE.findall(text)
        for i, part in enumerate(parts):
            if part:
                if self.sp is not None:
                    ids.extend(self.sp.encode(part, out_type=int))
                else:
                    ids.extend(list(part.encode("utf-8")))
            if i < len(specials):
                ids.append(SPECIAL_LOGIC_TOKENS[specials[i]])
        return ids

    def decode(self, ids: list[int]) -> str:
        """list of token ids -> text (special logic tokens mapped back to strings)."""
        # Fast path — no special IDs present
        if not any(tid >= 16384 for tid in ids):
            if self.sp is not None:
                return self.sp.decode(ids)
            return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")

        # Segment contiguous runs of BPE ids vs. special ids
        result_parts: list[str] = []
        bpe_buf: list[int] = []
        for tid in ids:
            if tid in _ID_TO_SPECIAL:
                if bpe_buf:
                    if self.sp is not None:
                        result_parts.append(self.sp.decode(bpe_buf))
                    else:
                        result_parts.append(bytes([b % 256 for b in bpe_buf]).decode("utf-8", errors="replace"))
                    bpe_buf = []
                result_parts.append(_ID_TO_SPECIAL[tid])
            else:
                bpe_buf.append(tid)
        if bpe_buf:
            if self.sp is not None:
                result_parts.append(self.sp.decode(bpe_buf))
            else:
                result_parts.append(bytes([b % 256 for b in bpe_buf]).decode("utf-8", errors="replace"))
        return "".join(result_parts)

    def encode_as_pieces(self, text: str) -> list[str]:
        """Text -> list of piece strings (display helper, e.g. for heatmaps)."""
        if self.sp is not None:
            return self.sp.encode(text, out_type=str)
        return list(text)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including dedicated logic tokens."""
        base = self.sp.get_piece_size() if self.sp is not None else 32768
        return base + NUM_SPECIAL_LOGIC

    @property
    def base_vocab_size(self) -> int:
        """Original BPE vocabulary size (without logic tokens)."""
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

    # -- Convenience accessors for dedicated logic token IDs --
    @property
    def cot_start_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<COT_START>"]

    @property
    def cot_end_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<COT_END>"]

    @property
    def rel_gt_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<REL_GT>"]

    @property
    def rel_lt_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<REL_LT>"]

    @property
    def rel_eq_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<REL_EQ>"]

    @property
    def rel_disjoint_id(self) -> int:
        return SPECIAL_LOGIC_TOKENS["<REL_DISJOINT>"]


def load_tokenizer(lang_dir: str, name: str = "") -> Tokenizer:
    """Load <lang_dir>/tokenizer/<lang>.model (name defaults to the dir basename)."""
    from pathlib import Path

    root = Path(lang_dir)
    lang = root.name
    model = root / "tokenizer" / (name or lang + ".model")
    return Tokenizer(str(model))
