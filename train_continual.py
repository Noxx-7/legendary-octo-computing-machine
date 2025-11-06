#!/usr/bin/env python3
"""
Continual Learning SLM Trainer
Trains the model sequentially on multiple datasets for progressive improvement
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json
import gc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import warnings
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import time
warnings.filterwarnings('ignore')

# Memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("="*80)
print("🚀 CONTINUAL LEARNING SLM TRAINER")
print("   Progressive training on multiple datasets")
print("="*80)

# ============ GPU Setup ============
def check_gpu():
    """Check for GPU availability and setup"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"   Total Memory: {total_memory:.2f} GB")
        
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        return True
    else:
        print("⚠️ No GPU available. Using CPU (very slow)")
        return False

USE_GPU = check_gpu()
device = torch.device('cuda' if USE_GPU else 'cpu')
print(f"🔧 Using device: {device}")

def clear_memory():
    """Clear GPU/CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============ Configuration ============
@dataclass
class ModelConfig:
    """Configuration for 125M parameter model"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072
    max_seq_len: int = 512
    batch_size: int = 8 if USE_GPU else 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-4
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = USE_GPU
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000

# ============ Dataset Configuration ============
DATASET_CONFIGS = [
    {
        'name': 'tinystories',
        'display_name': 'TinyStories',
        'dataset_path': 'roneneldan/TinyStories',
        'max_samples': 100000,  # Start small
        'epochs': 1,
        'description': 'Simple stories for foundational learning'
    },
    {
        'name': 'wikipedia',
        'display_name': 'Wikipedia',
        'dataset_path': ('wikipedia', '20220301.en'),
        'max_samples': 200000,
        'epochs': 1,
        'description': 'Factual knowledge and formal writing'
    },
    {
        'name': 'bookcorpus',
        'display_name': 'BookCorpus',
        'dataset_path': 'bookcorpusopen',
        'max_samples': 200000,
        'epochs': 1,
        'description': 'Literary style and narrative structure'
    },
    {
        'name': 'openwebtext',
        'display_name': 'OpenWebText',
        'dataset_path': 'Skylion007/openwebtext',
        'max_samples': 300000,
        'epochs': 1,
        'description': 'High-quality web content'
    },
    {
        'name': 'c4',
        'display_name': 'C4',
        'dataset_path': ('c4', 'en'),
        'max_samples': 500000,
        'epochs': 1,
        'description': 'Massive diverse web crawl'
    }
]

# ============ Dataset Loading ============
def load_dataset_for_continual(config_dict):
    """Load a specific dataset for continual learning"""
    name = config_dict['name']
    display = config_dict['display_name']
    path = config_dict['dataset_path']
    max_samples = config_dict['max_samples']
    
    print(f"\n📚 Loading {display} dataset...")
    print(f"   Description: {config_dict['description']}")
    
    try:
        if isinstance(path, tuple):
            # For datasets with configs like Wikipedia and C4
            dataset_name, config_name = path
            split = f"train[:{max_samples}]" if max_samples else "train"
            dataset = load_dataset(dataset_name, config_name, split=split)
            val_split = f"train[-5000:]"
            val_dataset = load_dataset(dataset_name, config_name, split=val_split)
        else:
            # For simple datasets
            split = f"train[:{max_samples}]" if max_samples else "train"
            dataset = load_dataset(path, split=split)
            
            # Handle validation split
            try:
                val_dataset = load_dataset(path, split="validation[:5000]")
            except:
                val_dataset = load_dataset(path, split="train[-5000:]")
        
        # Handle different text field names
        if name == 'wikipedia':
            dataset = dataset.map(lambda x: {"text": x.get("text", x.get("article", ""))})
            val_dataset = val_dataset.map(lambda x: {"text": x.get("text", x.get("article", ""))})
        elif name == 'bookcorpus':
            dataset = dataset.map(lambda x: {"text": x.get("text", x.get("title", ""))})
            val_dataset = val_dataset.map(lambda x: {"text": x.get("text", x.get("title", ""))})
        
        print(f"✅ Loaded {len(dataset):,} training samples")
        print(f"✅ Loaded {len(val_dataset):,} validation samples")
        
        return dataset, val_dataset
        
    except Exception as e:
        print(f"⚠️ Error loading {display}: {e}")
        print(f"   Skipping this dataset...")
        return None, None

# ============ Model Architecture ============
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))
    
    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Fix: Expand dimensions
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        cos, sin = self.rope(x, L)
        cos = cos.unsqueeze(0).unsqueeze(2)  
        sin = sin.unsqueeze(0).unsqueeze(2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is None:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Fix: FP16 compatibility
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(mask, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.ff_dim, bias=False)
        self.w2 = nn.Linear(config.ff_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.ff_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = RMSNorm(config.hidden_size)
        self.ln2 = RMSNorm(config.hidden_size)
    
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x

class SmallLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        B, L = input_ids.shape
        mask = torch.triu(torch.ones(L, L, device=input_ids.device), diagonal=1).bool()
        
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=0.8, top_p=0.9):
        self.eval()
        
        for _ in range(max_length - input_ids.shape[1]):
            logits, _ = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == 50256:  # EOS token
                break
        
        return input_ids

# ============ Dataset Preparation ============
class TextDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        labels = input_ids.clone()
        return input_ids, labels

# ============ Training Functions ============
def train_epoch(model, loader, optimizer, scheduler, scaler, config, epoch_num, dataset_name):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc=f"Training {dataset_name} - Epoch {epoch_num}")
    
    for i, (input_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        if config.use_mixed_precision:
            with autocast():
                logits, loss = model(input_ids, labels)
                loss = loss / config.gradient_accumulation_steps
        else:
            logits, loss = model(input_ids, labels)
            loss = loss / config.gradient_accumulation_steps
        
        if config.use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % config.gradient_accumulation_steps == 0:
            if config.use_mixed_precision:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            if config.use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        if i % config.log_interval == 0:
            avg_loss = total_loss / (i + 1)
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'ppl': f'{math.exp(min(avg_loss, 10)):.2f}'
            })
        
        if i % 100 == 0:
            clear_memory()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, config):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    for input_ids, labels in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        logits, loss = model(input_ids, labels)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    perplexity = math.exp(min(avg_loss, 10))
    
    return avg_loss, perplexity

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_length=100):
    """Generate text samples"""
    model.eval()
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=max_length)
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"💬 Generated: {generated_text}")

# ============ Continual Learning Training Loop ============
def continual_learning_train():
    """Main continual learning training loop"""
    config = ModelConfig()
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize or load model
    checkpoint_path = "continual_checkpoint.pt"
    start_dataset_idx = 0
    
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = SmallLanguageModel(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_dataset_idx = checkpoint.get('next_dataset_idx', 0)
        training_history = checkpoint.get('training_history', [])
        print(f"✅ Resumed from dataset index {start_dataset_idx}")
    else:
        print("\n🆕 Initializing new model...")
        model = SmallLanguageModel(config).to(device)
        training_history = []
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params/1e6:.1f}M")
    
    # Test generation prompts
    test_prompts = ["Once upon a time", "The future of AI is", "In the beginning"]
    
    # Train on each dataset sequentially
    for dataset_idx in range(start_dataset_idx, len(DATASET_CONFIGS)):
        dataset_config = DATASET_CONFIGS[dataset_idx]
        dataset_name = dataset_config['display_name']
        
        print("\n" + "="*80)
        print(f"📚 DATASET {dataset_idx + 1}/{len(DATASET_CONFIGS)}: {dataset_name}")
        print("="*80)
        
        # Load dataset
        train_dataset, val_dataset = load_dataset_for_continual(dataset_config)
        if train_dataset is None:
            continue
        
        # Tokenize dataset
        print("⚙️ Tokenizing dataset...")
        def tokenize_function(examples):
            text_field = 'text' if 'text' in examples else 'story'
            return tokenizer(
                examples[text_field],
                truncation=True,
                padding='max_length',
                max_length=config.max_seq_len,
                return_tensors=None
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True, 
                                           remove_columns=train_dataset.column_names)
        tokenized_val = val_dataset.map(tokenize_function, batched=True,
                                       remove_columns=val_dataset.column_names)
        
        # Create data loaders
        train_data = TextDataset(tokenized_train)
        val_data = TextDataset(tokenized_val)
        
        train_loader = DataLoader(train_data, batch_size=config.batch_size, 
                                shuffle=True, num_workers=2 if USE_GPU else 0)
        val_loader = DataLoader(val_data, batch_size=config.batch_size * 2,
                              shuffle=False, num_workers=2 if USE_GPU else 0)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training batches: {len(train_loader):,}")
        print(f"   Validation batches: {len(val_loader):,}")
        print(f"   Tokens: ~{len(train_loader) * config.batch_size * config.max_seq_len:,}")
        
        # Setup optimizer and scheduler for this dataset
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate * (0.8 ** dataset_idx),  # Decay LR for later datasets
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        total_steps = len(train_loader) * dataset_config['epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training epochs for this dataset
        dataset_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(1, dataset_config['epochs'] + 1):
            print(f"\n📅 Epoch {epoch}/{dataset_config['epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                   scaler, config, epoch, dataset_name)
            
            # Evaluate
            val_loss, val_ppl = evaluate(model, val_loader, config)
            
            dataset_losses.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl
            })
            
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_loss:.4f} | PPL: {math.exp(min(train_loss, 10)):.2f}")
            print(f"   Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
            
            # Save best model for this dataset
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'dataset_name': dataset_name,
                    'val_loss': val_loss
                }, f"best_model_{dataset_config['name']}.pt")
                print(f"   💾 Saved best model for {dataset_name}")
        
        # Generate samples after training on this dataset
        print(f"\n🔮 Generating samples after {dataset_name} training:")
        generate_samples(model, tokenizer, test_prompts, max_length=50)
        
        # Update training history
        training_history.append({
            'dataset': dataset_name,
            'losses': dataset_losses,
            'best_val_loss': best_val_loss
        })
        
        # Save continual learning checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'next_dataset_idx': dataset_idx + 1,
            'training_history': training_history
        }, checkpoint_path)
        print(f"\n💾 Saved continual learning checkpoint")
        
        clear_memory()
    
    # Final evaluation and summary
    print("\n" + "="*80)
    print("🎉 CONTINUAL LEARNING COMPLETE!")
    print("="*80)
    
    print("\n📊 Training Summary:")
    for dataset_history in training_history:
        print(f"\n{dataset_history['dataset']}:")
        print(f"   Best Val Loss: {dataset_history['best_val_loss']:.4f}")
        print(f"   Best Val PPL: {math.exp(min(dataset_history['best_val_loss'], 10)):.2f}")
    
    # Final generation examples
    print("\n🔮 Final Model Generation Examples:")
    final_prompts = [
        "Once upon a time, there was a",
        "The most important discovery in science is",
        "In the year 2050, technology will",
        "The secret to happiness is",
        "Artificial intelligence can help us"
    ]
    generate_samples(model, tokenizer, final_prompts, max_length=100)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_history
    }, 'final_continual_model.pt')
    print("\n💾 Saved final model to 'final_continual_model.pt'")

# ============ Main ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continual Learning SLM Trainer')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='Test generation only')
    args = parser.parse_args()
    
    if args.test_only:
        # Quick test
        print("\n🔮 Testing model generation...")
        config = ModelConfig()
        model = SmallLanguageModel(config).to(device)
        
        if os.path.exists('continual_checkpoint.pt'):
            checkpoint = torch.load('continual_checkpoint.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded checkpoint")
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        test_prompts = ["Once upon a time", "The future is", "AI will"]
        generate_samples(model, tokenizer, test_prompts, max_length=50)
    else:
        # Run continual learning
        continual_learning_train()