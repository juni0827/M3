from __future__ import annotations

from typing import List, Optional, Tuple

from llm_adapter.config import TokenizerConfig, get_global_config


class ByteTokenizer:
    """Byte-level tokenizer with special tokens PAD/BOS/EOS."""
    def __init__(self, config: Optional[TokenizerConfig] = None):
        cfg = config or get_global_config().tokenizer
        self.PAD = cfg.pad_id
        self.BOS = cfg.bos_id
        self.EOS = cfg.eos_id
        self.vocab_size = cfg.eos_id + 1  # 259 by default

    def encode(self, text: str, add_special: bool = False, max_len: Optional[int] = None) -> List[int]:
        b = text.encode('utf-8', errors='replace')
        ids = list(b)
        if add_special:
            ids = [self.BOS] + ids + [self.EOS]
        if max_len is not None:
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids += [self.PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        out: List[int] = []
        for t in ids:
            if t == self.PAD or t == self.EOS:
                break
            if t == self.BOS:
                continue
            if 0 <= int(t) <= 255:
                out.append(int(t))
        return bytes(out).decode('utf-8', errors='replace')


class HybridTokenizer:
    """
    Byte tokenizer + small learned merges (BPE-lite).
    Keeps PAD/BOS/EOS at same ids (from config) and adds merges at the tail.
    """
    def __init__(self, base: ByteTokenizer, merges: Optional[List[Tuple[bytes, bytes]]] = None, config: Optional[TokenizerConfig] = None):
        self.base = base
        cfg = config or get_global_config().tokenizer
        self.merges = merges or []  # list of (b1, b2)
        self.extra_vocab = int(cfg.extra_vocab)
        # build merge table and id map
        self._merge2id = {}
        start_id = base.vocab_size  # 259 by default
        for i, (a, b) in enumerate(self.merges[:self.extra_vocab]):
            self._merge2id[(a, b)] = start_id + i
        self.vocab_size = start_id + len(self._merge2id)

        # === Expose special tokens from base tokenizer ===
        self.PAD = base.PAD
        self.BOS = base.BOS
        self.EOS = base.EOS

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        raw = text.encode("utf-8", "ignore")
        toks = [c for c in raw]  # base bytes
        # greedy bigram merge
        i = 0
        out = []
        while i < len(toks):
            if i + 1 < len(toks):
                pair = (bytes([toks[i]]), bytes([toks[i + 1]]))
                mid = self._merge2id.get(pair)
                if mid is not None:
                    out.append(mid)
                    i += 2
                    continue
            out.append(toks[i])
            i += 1
        if add_special:
            out = [self.base.BOS] + out + [self.base.EOS]
        return out

    def decode(self, ids: List[int]) -> str:
        b = bytearray()
        inv = {v: k for k, v in self._merge2id.items()}
        for i in ids:
            if i in (self.base.PAD, self.base.BOS, self.base.EOS):
                continue
            if i < self.base.vocab_size:
                b.append(i)
            else:
                a, bg = inv.get(i, (b"", b""))
                b.extend(a)
                b.extend(bg)
        try:
            return b.decode("utf-8", "ignore")
        except Exception:
            return str(b)
