from __future__ import annotations

import os
import logging
from typing import List, Optional, Union

from llm_adapter.config import TokenizerConfig, get_global_config

logger = logging.getLogger(__name__)

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
        
        # Special tokens (M3 Standard)
        self.special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|affect|>", "<|drive|>", "<|memory|>", "<|user|>", "<|system|>"]

        # 1. Try HuggingFace tokenizers (Preferred)
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.trainers import BpeTrainer
            
            if vocab_file and os.path.exists(vocab_file):
                logger.info(f"Loading tokenizer from {vocab_file}")
                self._backend = Tokenizer.from_file(vocab_file)
                self._type = "hf"
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
                 else:
                     # If no vocab file, do NOT use empty BPE. Fallback to byte encoding or tiktoken.
                     # An empty BPE tokenizer cannot encode anything.
                     logger.warning("No vocab file found. Skipping HF BPE initialization to avoid empty vocabulary.")
                     raise ImportError("Force fallback")
            
            self._backend.add_special_tokens(self.special_tokens)
            
            self.pad_id = self._backend.token_to_id("<|pad|>")
            self.bos_id = self._backend.token_to_id("<|bos|>")
            self.eos_id = self._backend.token_to_id("<|eos|>")
            return
        except ImportError:
            pass

        # 2. Fallback to Tiktoken
        try:
            import tiktoken
            logger.warning("HuggingFace 'tokenizers' not found. Falling back to tiktoken.")
            self._backend = tiktoken.get_encoding("cl100k_base")
            self._type = "tiktoken"
            
            # Virtual mapping for special tokens
            base_vocab = self._backend.n_vocab
            self._special_ids = {t: base_vocab + i for i, t in enumerate(self.special_tokens)}
            self.pad_id = self._special_ids["<|pad|>"]
            self.bos_id = self._special_ids["<|bos|>"]
            self.eos_id = self._special_ids["<|eos|>"]
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
