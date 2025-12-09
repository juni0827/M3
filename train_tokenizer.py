
import os
import glob
from llm_adapter.tokenization import M3Tokenizer

def train_tokenizer():
    print("Starting tokenizer training...")
    
    # Initialize tokenizer (will fallback to tiktoken/byte initially, but we will train the HF backend)
    # We force HF backend initialization for training
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        
        backend = Tokenizer(BPE())
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        backend.decoder = ByteLevelDecoder()
        
        # Create a temporary M3Tokenizer to use its train method, 
        # but we need to inject the backend manually because __init__ might fail or fallback
        tok = M3Tokenizer()
        tok._backend = backend
        tok._type = "hf"
        
    except ImportError:
        print("Error: HuggingFace tokenizers library not found. Cannot train BPE tokenizer.")
        return

    # Find training files
    data_dir = 'data_set'
    files = []
    patterns = ['**/*.txt', '**/*.jsonl', '**/*.json', '**/*.tsv', '**/*.csv']
    
    for pattern in patterns:
        found = glob.glob(os.path.join(data_dir, pattern), recursive=True)
        for f in found:
            if os.path.isfile(f):
                # Skip files larger than 10MB to avoid memory issues and slow training
                if os.path.getsize(f) > 10 * 1024 * 1024:
                    continue
                files.append(f)
    
    print(f"Found {len(files)} files for training (filtered < 10MB).")
    
    # Limit files to avoid issues and speed up
    if len(files) > 100:
        print("Too many files, selecting random 100...")
        import random
        random.shuffle(files)
        files = files[:100]
        
    print(f"Selected files: {files[:5]}...")
    
    if not files:
        print("No files found. Aborting.")
        return

    # Train
    save_path = os.path.join('out_m3', 'tokenizer.json')
    os.makedirs('out_m3', exist_ok=True)
    
    print(f"Training on {len(files)} files...")
    try:
        tok.train(files, vocab_size=30000)
        tok.save(save_path)
        print(f"Tokenizer saved to {save_path}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("DONE")

if __name__ == "__main__":
    train_tokenizer()
