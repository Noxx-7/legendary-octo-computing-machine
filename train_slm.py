#!/usr/bin/env python3
"""
Small Language Model Training Script for T4 GPU
Optimized for 15GB memory with ~125M parameters
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import gc
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
warnings.filterwarnings('ignore')

# Memory optimizations for T4
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@dataclass
class ModelConfig:
    """Configuration for 125M parameter model optimized for T4"""
    
    # Model architecture (125M parameters)
    vocab_size: int = 32000  
    hidden_size: int = 768   
    num_layers: int = 12     
    num_heads: int = 12      
    ff_dim: int = 3072       
    max_seq_len: int = 512   
    
    # Training configuration
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 6e-4
    num_epochs: int = 1  # Quick demo
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    weight_decay: float = 0.01
    
    # Optimization flags
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    compile_model: bool = False
    
    # Checkpointing
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    logging_steps: int = 10
    
    # Paths
    checkpoint_dir: str = "./checkpoints"

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

def clear_memory():
    """Clear GPU/CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
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
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # Repeat cos and sin to match full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""
    
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
        
        # Fix RoPE dimensions
        cos, sin = self.rope(x, L)
        # Reshape for broadcasting: [seq_len, dim//2] -> [1, seq_len, 1, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  
        sin = sin.unsqueeze(0).unsqueeze(2)
        # Now cos and sin have shape [1, L, 1, head_dim//2] which will broadcast correctly
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is None:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Use smaller value for FP16 compatibility (-1e9 overflows in half precision)
        mask_value = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(mask, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        
        return out

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.ff_dim, bias=False)
        self.w2 = nn.Linear(config.ff_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.ff_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
    
    def forward(self, x, mask=None):
        x = x + self.attention(self.ln1(x), mask)
        x = x + self.feed_forward(self.ln2(x))
        return x

class SmallLanguageModel(nn.Module):
    """Complete Small Language Model for T4 GPU"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n✅ Model initialized:")
        print(f"   Total parameters: {total_params/1e6:.1f}M")
        print(f"   Memory footprint: ~{total_params * 4 / 1e9:.2f} GB (FP32)")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        mask = torch.triu(torch.ones(L, L, device=input_ids.device), diagonal=1).bool()
        
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_p=0.95):
        """Generate text using the model"""
        self.eval()
        
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if input_ids.shape[1] >= self.config.max_seq_len:
                break
        
        return input_ids

class TextDataset(Dataset):
    """Dataset for loading and processing text data"""
    
    def __init__(self, encodings, seq_len):
        self.encodings = encodings
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

def load_and_prepare_data(config):
    """Load dataset from HuggingFace and prepare for training"""
    print("\n📚 Loading dataset...")
    
    # Load a subset for demo - use larger dataset for real training
    dataset = load_dataset('roneneldan/TinyStories', split='train[:5000]')
    print(f"   Loaded {len(dataset)} examples")
    
    print("\n🔤 Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n⚙️ Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_len,
            return_tensors='pt'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = TextDataset(
        {'input_ids': tokenized_dataset['input_ids'][:train_size]},
        config.max_seq_len
    )
    val_dataset = TextDataset(
        {'input_ids': tokenized_dataset['input_ids'][train_size:]},
        config.max_seq_len
    )
    
    print(f"\n✅ Data prepared:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    config.vocab_size = tokenizer.vocab_size
    
    return train_dataset, val_dataset, tokenizer

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits, loss = model(input_ids, labels)
        
        total_loss += loss.item() * input_ids.shape[0]
        total_tokens += input_ids.shape[0]
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    
    model.train()
    return avg_loss, perplexity

def save_checkpoint(model, optimizer, epoch, step, loss, config):
    """Save model checkpoint"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{step}.pt')
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__
    }, checkpoint_path)
    
    print(f"\n💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, config, tokenizer, device):
    """Main training loop"""
    
    print("\n" + "="*80)
    print("🚀 Starting training...")
    print("="*80)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{config.num_epochs}")
        
        model.train()
        accumulated_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
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
            
            accumulated_loss += loss.item()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    avg_loss = accumulated_loss * config.gradient_accumulation_steps
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    accumulated_loss = 0
                
                if global_step % config.eval_every_n_steps == 0:
                    print(f"\n📊 Evaluating at step {global_step}...")
                    val_loss, val_perplexity = evaluate(model, val_loader, device)
                    print(f"   Validation Loss: {val_loss:.4f}")
                    print(f"   Validation Perplexity: {val_perplexity:.2f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, epoch, global_step, val_loss, config)
                
                if global_step % config.save_every_n_steps == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, accumulated_loss, config)
            
            if step % 100 == 0:
                clear_memory()
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Train a Small Language Model on T4 GPU')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--test-only', action='store_true', help='Run quick test only')
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 T4 GPU OPTIMIZED 125M PARAMETER MODEL")
    print("="*80)
    
    # Check GPU
    USE_GPU = check_gpu()
    device = torch.device('cuda' if USE_GPU else 'cpu')
    print(f"🔧 Using device: {device}")
    clear_memory()
    
    # Create configuration
    config = ModelConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.use_mixed_precision = USE_GPU
    
    # Load data
    train_dataset, val_dataset, tokenizer = load_and_prepare_data(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=USE_GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=USE_GPU
    )
    
    # Create model
    model = SmallLanguageModel(config).to(device)
    
    # Test generation before training
    if args.test_only:
        print("\n🔮 Testing generation (untrained model)...")
        tokenizer_ids = tokenizer("Once upon a time", return_tensors='pt')['input_ids'].to(device)
        generated = model.generate(tokenizer_ids, max_new_tokens=20)
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {output}")
        print("\n✅ Test complete! Model is working.")
        return
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Create gradient scaler
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return current_step / config.warmup_steps
        progress = (current_step - config.warmup_steps) / (num_training_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, config, tokenizer, device)
    
    # Final generation test
    print("\n🔮 Testing generation after training...")
    test_prompts = [
        "Once upon a time,",
        "The little girl",
        "In a magical forest,"
    ]
    
    for prompt in test_prompts:
        tokenizer_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        generated = model.generate(tokenizer_ids, max_new_tokens=30)
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output}")
    
    print("\n✨ All done! Happy modeling! ✨")

if __name__ == "__main__":
    main()