from __future__ import annotations

import json
import os
import logging
import time
import re
import hashlib
from typing import Iterable, List, Optional, Dict, Any, Set

from llm_adapter.config import TokenizerConfig, get_global_config

logger = logging.getLogger(__name__)
_LOG_ONCE_KEYS: Set[str] = set()


def _log_once(level: str, message: str, *args: Any) -> None:
    try:
        rendered = message % args if args else message
    except Exception:
        rendered = message
    key = f"{level}:{rendered}"
    if key in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(key)
    getattr(logger, level)(message, *args)

class M3Tokenizer:
    """
    The definitive M3 Tokenizer.
    Leverages the high-performance HuggingFace 'tokenizers' library (Rust) for BPE.
    Supports training, saving, and special token handling for M3 architecture.
    Falls back to Tiktoken or Byte encoding if necessary, but encapsulated within this single class.
    """
    def __init__(self, vocab_file: Optional[str] = None, config: Optional[TokenizerConfig] = None):
        self.config = config or get_global_config().tokenizer
        self._backend = None
        self._type = "unknown"
        self._unknown_tokens = 0
        self._observed_tokens = 0
        self._unknown_observations = 0
        self._unknown_rate_ema = 0.0
        self._last_rebuild_observation = -10**9
        self._last_rebuild_path = ""
        self._last_rebuild_unix = 0.0
        self._rebuild_count = 0
        self._last_rebuild_corpus_fingerprint = ""
        
        # Special tokens (M3 Standard)
        self.special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|affect|>", "<|drive|>", "<|memory|>", "<|user|>", "<|system|>"]

        # 1. Try HuggingFace tokenizers (Preferred)
        hf_available = False
        hf_loaded = False
        hf_reason = ""
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.trainers import BpeTrainer
            hf_available = True
            
            if vocab_file and os.path.exists(vocab_file):
                logger.info(f"Loading tokenizer from {vocab_file}")
                self._backend = Tokenizer.from_file(vocab_file)
                self._type = "hf"
                hf_loaded = True
            else:
                 # Check unified logs path
                 default_path = os.path.join('docs&tests&data_sets', 'tests', 'logs', 'tokenizer.json')
                 # Check legacy path
                 legacy_path = os.path.join('out_m3', 'tokenizer.json')
                 
                 found_path = None
                 if os.path.exists(default_path):
                     found_path = default_path
                 elif os.path.exists(legacy_path):
                     found_path = legacy_path
                     
                 if found_path:
                     logger.info(f"Loading tokenizer from default path {found_path}")
                     self._backend = Tokenizer.from_file(found_path)
                     self._type = "hf"
                     hf_loaded = True
                 else:
                     # No vocab file yet: fallback to tiktoken/byte until a rebuild provides one.
                     hf_reason = "missing_vocab"
                     _log_once(
                         "info",
                         "No vocab file found. Skipping HF BPE initialization to avoid empty vocabulary.",
                     )
            
            if hf_loaded:
                self._backend.add_special_tokens(self.special_tokens)
                
                self.pad_id = self._backend.token_to_id("<|pad|>")
                self.bos_id = self._backend.token_to_id("<|bos|>")
                self.eos_id = self._backend.token_to_id("<|eos|>")
                self.load_rebuild_state()
                return
        except ImportError:
            hf_reason = "missing_tokenizers_pkg"
            pass
        except Exception as e:
            hf_reason = f"hf_init_error:{e}"

        # 2. Fallback to Tiktoken
        try:
            import tiktoken
            if hf_reason == "missing_tokenizers_pkg":
                _log_once("warning", "HuggingFace 'tokenizers' not found. Falling back to tiktoken.")
            elif hf_reason == "missing_vocab":
                _log_once("info", "HF tokenizer package is available, but tokenizer vocab file is missing. Falling back to tiktoken.")
            elif hf_reason.startswith("hf_init_error:"):
                _log_once("warning", "HF tokenizer initialization failed (%s). Falling back to tiktoken.", hf_reason.split(":", 1)[1])
            elif not hf_available:
                _log_once("warning", "HF tokenizer unavailable. Falling back to tiktoken.")
            self._backend = tiktoken.get_encoding("cl100k_base")
            self._type = "tiktoken"
            
            # Virtual mapping for special tokens
            base_vocab = self._backend.n_vocab
            self._special_ids = {t: base_vocab + i for i, t in enumerate(self.special_tokens)}
            self.pad_id = self._special_ids["<|pad|>"]
            self.bos_id = self._special_ids["<|bos|>"]
            self.eos_id = self._special_ids["<|eos|>"]
            self.load_rebuild_state()
            return
        except ImportError:
            pass

        # 3. Fallback to simple Byte encoding
        logger.warning("No high-performance tokenizer found. Using slow Byte fallback.")
        self._type = "byte"
        # Virtual mapping
        self._special_ids = {t: 256 + i for i, t in enumerate(self.special_tokens)}
        self.pad_id = self._special_ids["<|pad|>"]
        self.bos_id = self._special_ids["<|bos|>"]
        self.eos_id = self._special_ids["<|eos|>"]
        self.load_rebuild_state()

    @property
    def PAD(self) -> int:
        return self.pad_id

    @property
    def BOS(self) -> int:
        return self.bos_id

    @property
    def EOS(self) -> int:
        return self.eos_id

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = []
        if self._type == "hf":
            ids = self._backend.encode(text).ids
        elif self._type == "tiktoken":
            try:
                ids = self._backend.encode(text)
            except:
                ids = []
        else: # byte
            ids = list(text.encode('utf-8', errors='replace'))

        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        try:
            self.observe_unknown_rate(text)
        except Exception:
            pass
        return ids

    def decode(self, ids: List[int]) -> str:
        if self._type == "hf":
            return self._backend.decode(ids, skip_special_tokens=True)
        elif self._type == "tiktoken":
            # Filter special tokens
            clean_ids = [i for i in ids if i < self._backend.n_vocab]
            return self._backend.decode(clean_ids)
        else: # byte
            clean_ids = [i for i in ids if i < 256]
            return bytes(clean_ids).decode('utf-8', errors='replace')

    @property
    def vocab_size(self) -> int:
        if self._type == "hf":
            return self._backend.get_vocab_size()
        elif self._type == "tiktoken":
            return self._backend.n_vocab + len(self.special_tokens)
        else:
            return 256 + len(self.special_tokens)

    def train(self, files: List[str], vocab_size: int = 30000):
        """Train the tokenizer (Only supported for HF backend)."""
        if self._type != "hf":
            logger.warning("Training is only supported for HuggingFace backend.")
            return
            
        from tokenizers.trainers import BpeTrainer
        logger.info(f"Training tokenizer on {len(files)} files...")
        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )

        def safe_file_iterator(file_paths):
            for path in file_paths:
                try:
                    # Attempt to read with utf-8 and ignore errors to skip bad bytes
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        yield f.read()
                except Exception as e:
                    logger.warning(f"Failed to read file {path}: {e}")
                    continue

        # Use train_from_iterator instead of train(files) to handle encoding safely in Python
        self._backend.train_from_iterator(safe_file_iterator(files), trainer=trainer, length=len(files))
        logger.info("Tokenizer training completed.")

    def _safe_text_iter(self, files: List[str], max_chars: int = 1_500_000) -> Iterable[str]:
        total_chars = 0
        for path in files:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    if path.lower().endswith('.jsonl'):
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                if isinstance(obj, dict):
                                    for key in ("prompt", "response", "chosen", "rejected", "content", "text", "input", "output"):
                                        val = obj.get(key)
                                        if isinstance(val, str) and val.strip():
                                            seg = val.strip()
                                            total_chars += len(seg)
                                            yield seg
                                            if max_chars > 0 and total_chars >= max_chars:
                                                return
                            except Exception:
                                total_chars += len(line)
                                yield line
                                if max_chars > 0 and total_chars >= max_chars:
                                    return
                    else:
                        data = f.read()
                        if not data:
                            continue
                        total_chars += len(data)
                        yield data
                        if max_chars > 0 and total_chars >= max_chars:
                            return
            except Exception:
                continue

    def _rebuild_state_path(self) -> str:
        cfg = get_global_config().tokenizer_auto_vocab
        p = str(os.getenv("M3_TOKENIZER_REBUILD_STATE", cfg.state_file or "tokenizer_rebuild_state.json"))
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        return p

    def load_rebuild_state(self) -> Dict[str, Any]:
        path = self._rebuild_state_path()
        try:
            if not os.path.exists(path):
                return {"ok": False, "reason": "missing", "path": path}
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
            self._last_rebuild_unix = float(st.get("last_rebuild_unix", 0.0))
            self._rebuild_count = int(st.get("rebuild_count", 0))
            self._last_rebuild_observation = int(st.get("last_rebuild_observation", self._last_rebuild_observation))
            self._last_rebuild_path = str(st.get("last_rebuild_path", self._last_rebuild_path or ""))
            self._last_rebuild_corpus_fingerprint = str(st.get("last_rebuild_corpus_fingerprint", ""))
            return {"ok": True, "path": path}
        except Exception as e:
            logger.warning(f"Failed to load tokenizer rebuild state: {e}")
            return {"ok": False, "reason": str(e), "path": path}

    def save_rebuild_state(self) -> Dict[str, Any]:
        path = self._rebuild_state_path()
        payload = {
            "last_rebuild_unix": float(self._last_rebuild_unix),
            "rebuild_count": int(self._rebuild_count),
            "last_rebuild_observation": int(self._last_rebuild_observation),
            "last_rebuild_path": str(self._last_rebuild_path or ""),
            "last_rebuild_corpus_fingerprint": str(self._last_rebuild_corpus_fingerprint or ""),
            "vocab_size": int(self.vocab_size),
        }
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return {"ok": True, "path": path}
        except Exception as e:
            logger.warning(f"Failed to save tokenizer rebuild state: {e}")
            return {"ok": False, "reason": str(e), "path": path}

    def export_vocab_snapshot(self) -> Dict[str, Any]:
        if self._type == "hf":
            try:
                vocab = self._backend.get_vocab()
                return {"ok": True, "type": "hf", "size": int(len(vocab)), "tokens": set(vocab.keys())}
            except Exception as e:
                return {"ok": False, "reason": str(e), "type": "hf", "size": int(self.vocab_size), "tokens": set()}
        return {"ok": True, "type": self._type, "size": int(self.vocab_size), "tokens": set()}

    def _collect_corpus_stats(self, files: List[str], max_chars: int) -> Dict[str, Any]:
        total_chars = 0
        total_docs = 0
        unique_terms: Set[str] = set()
        for seg in self._safe_text_iter(files, max_chars=max_chars):
            if not seg:
                continue
            total_docs += 1
            total_chars += len(seg)
            for term in re.findall(r"[A-Za-z0-9_]+|[\uac00-\ud7a3]+", seg.lower()):
                if len(term) >= 2:
                    unique_terms.add(term)
            if max_chars > 0 and total_chars >= max_chars:
                break
        return {
            "docs": int(total_docs),
            "chars": int(total_chars),
            "unique_terms": int(len(unique_terms)),
        }

    def can_rebuild_safely(
        self,
        corpus_stats: Dict[str, Any],
        old_vocab_size: int,
        new_vocab_size: Optional[int] = None,
    ) -> bool:
        cfg = get_global_config().tokenizer_auto_vocab
        try:
            chars = int(corpus_stats.get("chars", 0))
        except Exception:
            chars = 0
        try:
            uniq = int(corpus_stats.get("unique_terms", 0))
        except Exception:
            uniq = 0
        if chars < int(max(1, cfg.min_corpus_chars)):
            return False
        if uniq < int(max(1, cfg.min_unique_terms)):
            return False
        if new_vocab_size is not None:
            floor_ratio = float(max(1e-6, min(1.0, cfg.min_keep_vocab_ratio)))
            if int(new_vocab_size) < int(max(32, floor_ratio * max(1, int(old_vocab_size)))):
                return False
        return True

    def observe_unknown_rate(self, text: str) -> float:
        s = str(text or "")
        if not s:
            return float(self._unknown_rate_ema)
        unknown = 0
        total = 0
        if self._type == "hf":
            enc = self._backend.encode(s)
            ids = list(enc.ids)
            total = len(ids)
            unk_id = self._backend.token_to_id("<|unk|>")
            if unk_id is None:
                unk_id = self._backend.token_to_id("[UNK]")
            if unk_id is not None and total > 0:
                unknown = int(sum(1 for i in ids if int(i) == int(unk_id)))
        elif self._type == "byte":
            raw = s.encode('utf-8', errors='replace')
            total = len(raw)
            unknown = int(raw.count(b'?'))
        else:
            total = 0
            unknown = 0
        if total > 0:
            self._unknown_tokens += int(unknown)
            self._observed_tokens += int(total)
            self._unknown_observations += 1
            cur = float(unknown) / float(max(1, total))
            alpha = 0.05
            self._unknown_rate_ema = (1.0 - alpha) * float(self._unknown_rate_ema) + alpha * cur
        return float(self._unknown_rate_ema)

    def should_rebuild_vocab(self) -> bool:
        cfg = get_global_config().tokenizer_auto_vocab
        enabled = cfg.enabled and str(os.getenv("M3_TOKENIZER_AUTO_VOCAB", "1")).lower() in ("1", "true", "yes", "on")
        if not enabled:
            return False
        now = time.time()
        min_interval = int(max(0, cfg.rebuild_min_interval_sec))
        if min_interval > 0 and (now - float(self._last_rebuild_unix)) < float(min_interval):
            return False
        if self._type != "hf":
            # Rebuild can still create HF tokenizer from scratch, so allow.
            pass
        min_obs = int(max(1, cfg.min_observations))
        if int(self._unknown_observations) < min_obs:
            return False
        cooldown = int(max(1, cfg.cooldown_steps))
        if (int(self._unknown_observations) - int(self._last_rebuild_observation)) < cooldown:
            return False
        threshold = float(max(0.0, cfg.unknown_rate_threshold))
        global_rate = float(self._unknown_tokens) / float(max(1, self._observed_tokens))
        rate = max(global_rate, float(self._unknown_rate_ema))
        return bool(rate >= threshold)

    def rebuild_vocab_from_corpus(self, files: List[str], out_path: str, vocab_size: int) -> bool:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from tokenizers.trainers import BpeTrainer

        if not files:
            logger.warning("Tokenizer rebuild skipped: empty corpus files.")
            return False
        old_vocab_size = int(self.vocab_size)
        cfg = get_global_config().tokenizer_auto_vocab
        max_chars = int(max(1_000, cfg.corpus_max_chars))
        corpus_stats = self._collect_corpus_stats(files, max_chars=max_chars)
        if not self.can_rebuild_safely(corpus_stats, old_vocab_size=old_vocab_size):
            logger.warning(
                "Tokenizer rebuild guard: insufficient corpus (chars=%s unique_terms=%s)",
                corpus_stats.get("chars", 0),
                corpus_stats.get("unique_terms", 0),
            )
            return False
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        tok.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=int(max(256, vocab_size)),
            special_tokens=list(self.special_tokens),
            min_frequency=2,
            show_progress=False,
        )
        try:
            corpus_iter = self._safe_text_iter(files, max_chars=max_chars)
            tok.train_from_iterator(corpus_iter, trainer=trainer)
            new_vocab_size = int(tok.get_vocab_size())
            if not self.can_rebuild_safely(
                corpus_stats,
                old_vocab_size=old_vocab_size,
                new_vocab_size=new_vocab_size,
            ):
                logger.warning(
                    "Tokenizer rebuild guard: new vocab too small old=%d new=%d",
                    int(old_vocab_size),
                    int(new_vocab_size),
                )
                return False
            tok.save(out_path)
        except Exception as e:
            logger.warning(f"Tokenizer rebuild failed: {e}")
            return False
        self._backend = tok
        self._type = "hf"
        self._backend.add_special_tokens(self.special_tokens)
        self.pad_id = self._backend.token_to_id("<|pad|>")
        self.bos_id = self._backend.token_to_id("<|bos|>")
        self.eos_id = self._backend.token_to_id("<|eos|>")
        self._last_rebuild_observation = int(self._unknown_observations)
        self._last_rebuild_path = out_path
        self._last_rebuild_unix = float(time.time())
        self._rebuild_count = int(self._rebuild_count) + 1
        try:
            fp_payload = {
                "files": [str(p) for p in files],
                "stats": corpus_stats,
                "vocab_size": int(vocab_size),
            }
            self._last_rebuild_corpus_fingerprint = hashlib.sha1(
                json.dumps(fp_payload, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
        except Exception:
            self._last_rebuild_corpus_fingerprint = ""
        self._unknown_tokens = 0
        self._observed_tokens = 0
        self._unknown_rate_ema = 0.0
        self.save_rebuild_state()
        return True

    def save(self, path: str):
        if self._type == "hf":
            self._backend.save(path)
            logger.info(f"Tokenizer saved to {path}")
        else:
            logger.warning("Save not supported for this backend.")

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> 'M3Tokenizer':
        return cls(config=config)

# Alias for compatibility
AutoTokenizer = M3Tokenizer
