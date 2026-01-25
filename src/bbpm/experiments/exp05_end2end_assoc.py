"""Experiment 05: End-to-end associative recall."""

import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bbpm.addressing.hash_mix import mix64, u64
from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from bbpm.metrics.stats import mean_ci95
from bbpm.experiments.common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
    make_rng,
)
from bbpm.experiments.plotting import save_pdf, add_footer
from bbpm.utils.seeds import seed_everything

EXP_ID = "exp05"
EXP_SLUG = "end2end_assoc"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add experiment-specific arguments."""
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=64,
        help="Sequence length",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=100,  # Reduced from 1000 to make task learnable (chance = 1/100 = 0.01)
        help="Vocabulary size",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,  # Increased from 64 to increase model capacity
        help="Model dimension",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )


def generate_batch(
    batch_size: int,
    T: int,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate canonical associative recall task batch with token IDs.
    
    Task: Given sequence of (key, value) pairs, retrieve value token ID for a query key.
    Sequence format: [key1, val1, key2, val2, ..., query_key]
    Target: value token ID corresponding to query_key (classification).
    
    Args:
        batch_size: Batch size
        T: Sequence length (must be odd: num_pairs*2 + 1 for query)
        vocab_size: Vocabulary size
        seed: Random seed
        
    Returns:
        (sequences, targets)
        sequences: [batch_size, T] token IDs
        targets: [batch_size] target value token IDs
    """
    rng = make_rng(seed)
    
    sequences = []
    targets = []
    
    for _ in range(batch_size):
        # Generate pairs: (key, value) pairs
        num_pairs = (T - 1) // 2
        pairs = []
        used_keys = set()
        
        for _ in range(num_pairs):
            key_id = int(rng.integers(0, vocab_size))
            val_id = int(rng.integers(0, vocab_size))
            pairs.append((key_id, val_id))
            used_keys.add(key_id)
        
        # Query key: must appear in pairs
        query_key_id = rng.choice(list(used_keys))
        
        # Find target value token ID
        target_val_id = None
        for k, v in pairs:
            if k == query_key_id:
                target_val_id = v
                break
        
        # Build sequence: [key1, val1, key2, val2, ..., query_key]
        seq_token_ids = []
        for k, v in pairs:
            seq_token_ids.append(k)
            seq_token_ids.append(v)
        seq_token_ids.append(query_key_id)  # Query at end
        
        sequences.append(seq_token_ids)
        targets.append(target_val_id)
    
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class WindowedTransformer(nn.Module):
    """Windowed transformer baseline with shared embedding."""
    
    def __init__(self, embedding: nn.Embedding, d_model: int, num_heads: int, window_size: int, vocab_size: int):
        super().__init__()
        self.embedding = embedding
        self.window_size = window_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4),
            num_layers=2,
        )
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T] token IDs"""
        # Embed tokens
        x_emb = self.embedding(x)  # [batch, T, d_model]
        # Use last window_size tokens
        x_window = x_emb[:, -self.window_size:, :]  # [batch, window_size, d_model]
        # Transformer expects [seq_len, batch, d_model]
        x_window = x_window.transpose(0, 1)  # [window_size, batch, d_model]
        out = self.transformer(x_window)  # [window_size, batch, d_model]
        # Use last token: out[-1] is [batch, d_model]
        return self.classifier(out[-1])  # [batch, vocab_size]


class TransformerWithExternalKV(nn.Module):
    """Transformer with external learned KV memory and shared embedding."""
    
    def __init__(self, embedding: nn.Embedding, d_model: int, num_heads: int, memory_size: int, vocab_size: int):
        super().__init__()
        self.embedding = embedding
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4),
            num_layers=2,
        )
        self.attention_to_memory = nn.MultiheadAttention(d_model, num_heads)
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T] token IDs"""
        # Embed tokens
        x_emb = self.embedding(x)  # [batch, T, d_model]
        # Process sequence
        x_seq = x_emb.transpose(0, 1)  # [T, batch, d_model]
        x_encoded = self.transformer(x_seq)
        query = x_encoded[-1:]  # Last token [1, batch, d_model]
        
        # Attend to external memory
        memory = self.memory.unsqueeze(1).expand(-1, query.size(1), -1)  # [M, batch, d_model]
        attn_out, _ = self.attention_to_memory(query, memory, memory)
        return self.classifier(attn_out.squeeze(0))  # [batch, vocab_size]


class BBPMAssociativeModel(nn.Module):
    """BBPM-based associative recall model with token ID keying."""
    
    def __init__(self, embedding: nn.Embedding, d_model: int, mem_cfg: MemoryConfig, vocab_size: int):
        super().__init__()
        self.embedding = embedding
        self.mem_cfg = mem_cfg  # Store config, create memory per forward
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T] token IDs"""
        # Embed tokens
        x_emb = self.embedding(x)  # [batch, T, d_model]
        
        batch_size = x.shape[0]
        T = x.shape[1]
        
        # Create fresh memory instance for this forward pass to avoid graph issues
        mem = BBPMMemory(self.mem_cfg)
        
        # Process each sequence in batch
        retrieved_list = []
        for b in range(batch_size):
            # Reset memory for each sequence
            mem.reset()
            
            # Write (key, value) pairs: even indices are keys, odd are values
            for t in range(0, T - 1, 2):  # Step by 2 for pairs
                if t + 1 < T - 1:  # Ensure we have a pair
                    key_id = int(x[b, t].item())
                    val_id = int(x[b, t + 1].item())
                    val_emb = x_emb[b, t + 1]  # Keep gradients for learnable embeddings
                    
                    # Use token ID keying with master_seed for stability: hx = mix64(u64(token_id) ^ u64(master_seed))
                    hx = mix64(u64(key_id) ^ u64(self.mem_cfg.master_seed))
                    mem.write(hx, val_emb)
            
            # Retrieve using query (last token)
            query_id = int(x[b, -1].item())
            hx = mix64(u64(query_id) ^ u64(self.mem_cfg.master_seed))
            r = mem.read(hx)
            retrieved_list.append(r)
        
        retrieved_tensor = torch.stack(retrieved_list)  # [batch, d_model]
        return self.classifier(retrieved_tensor)  # [batch, vocab_size]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run end-to-end associative recall experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with metrics_path and figure_path
    """
    device = ensure_device(args.device)
    dtype_str = args.dtype
    num_seeds = args.seeds
    out_dir = args.out_dir
    
    # Task configuration
    batch_size = args.batch_size
    T = args.T
    vocab_size = args.vocab_size
    d_model = args.d_model
    num_heads = 4
    num_epochs = args.num_epochs
    lr = 1e-3  # Learning rate - may need tuning if models don't learn
    
    # Increase learning rate slightly for better learning
    # But keep it reasonable to avoid instability
    if vocab_size <= 100:
        lr = 2e-3  # Slightly higher LR for smaller vocab (easier task)
    
    # BBPM memory configuration
    B = 2**12  # 4096 blocks
    L = 256
    K = 32
    H = 4
    
    mem_cfg = MemoryConfig(
        num_blocks=B,
        block_size=L,
        key_dim=d_model,
        K=K,
        H=H,
        dtype=dtype_str,
        device=str(device),
        normalize_values="none",
        read_mode="raw_mean",
        master_seed=42,
    )
    
    seeds = seed_loop(num_seeds)
    raw_trials = []
    
    for seed in seeds:
        seed_everything(seed)
        
        # Create shared embedding layer
        # NOTE: Embeddings are trainable (not fixed) to allow models to learn token representations
        # This is a design choice for the associative recall task - embeddings learn to encode
        # semantic relationships between keys and values.
        embedding = nn.Embedding(vocab_size, d_model).to(device)
        # Embeddings are trainable by default - this allows models to learn better representations
        
        # Generate train/test data with separate RNG
        train_seqs, train_targets = generate_batch(
            batch_size * 10, T, vocab_size, seed
        )
        test_seqs, test_targets = generate_batch(
            batch_size * 5, T, vocab_size, seed + 1000
        )
        train_seqs = train_seqs.to(device)
        train_targets = train_targets.to(device)
        test_seqs = test_seqs.to(device)
        test_targets = test_targets.to(device)
        
        # Model 1: Windowed Transformer
        # Window size should be large enough to see at least one (key, value) pair + query
        # Sequence format: [key1, val1, key2, val2, ..., query_key]
        # So window_size should be >= 3 to see at least one pair + query
        # Use larger window to see more context
        window_size = min(32, T)  # Window size up to sequence length, but at least see query
        model1 = WindowedTransformer(embedding, d_model, num_heads, window_size, vocab_size).to(device)
        param_count1 = count_parameters(model1)
        
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses1 = []
        test_accs1 = []
        
        for epoch in range(num_epochs):
            # Training
            model1.train()
            epoch_loss = 0
            for i in range(0, len(train_seqs), batch_size):
                batch_seqs = train_seqs[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                optimizer1.zero_grad()
                outputs = model1(batch_seqs)  # [batch, vocab_size]
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer1.step()
                epoch_loss += loss.item()
            
            train_losses1.append(epoch_loss / (len(train_seqs) // batch_size))
            
            # Testing
            model1.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i in range(0, len(test_seqs), batch_size):
                    batch_seqs = test_seqs[i:i+batch_size]
                    batch_targets = test_targets[i:i+batch_size]
                    outputs = model1(batch_seqs)  # [batch, vocab_size]
                    # Accuracy: exact match (top-1)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == batch_targets).sum().item()
                    total += len(batch_targets)
                test_accs1.append(correct / total)
        
        # Model 2: Transformer + External KV
        # Adjust memory_size to match param count with BBPM
        # Target: roughly match model1 or model3 params
        target_params = param_count1
        # Approximate: embedding + transformer + memory + classifier
        # embedding: vocab_size * d_model
        # transformer: ~2 * (4 * d_model^2 + 2 * d_model * 4*d_model) = ~24 * d_model^2
        # memory: memory_size * d_model
        # classifier: d_model * vocab_size
        embedding_params = vocab_size * d_model
        transformer_params = 24 * d_model * d_model  # Approximate
        classifier_params = d_model * vocab_size
        # Solve for memory_size to match target
        memory_params_target = target_params - embedding_params - transformer_params - classifier_params
        memory_size = max(100, int(memory_params_target / d_model))
        
        model2 = TransformerWithExternalKV(embedding, d_model, num_heads, memory_size, vocab_size).to(device)
        param_count2 = count_parameters(model2)
        
        # Check parameter count difference
        param_diff_pct = abs(param_count2 - param_count1) / param_count1 * 100
        if param_diff_pct > 5:
            print(f"Warning: Model2 param count differs by {param_diff_pct:.1f}% from Model1")
        
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        
        train_losses2 = []
        test_accs2 = []
        
        for epoch in range(num_epochs):
            model2.train()
            epoch_loss = 0
            for i in range(0, len(train_seqs), batch_size):
                batch_seqs = train_seqs[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                optimizer2.zero_grad()
                outputs = model2(batch_seqs)  # [batch, vocab_size]
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer2.step()
                epoch_loss += loss.item()
            
            train_losses2.append(epoch_loss / (len(train_seqs) // batch_size))
            
            model2.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i in range(0, len(test_seqs), batch_size):
                    batch_seqs = test_seqs[i:i+batch_size]
                    batch_targets = test_targets[i:i+batch_size]
                    outputs = model2(batch_seqs)  # [batch, vocab_size]
                    # Accuracy: exact match (top-1)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == batch_targets).sum().item()
                    total += len(batch_targets)
                test_accs2.append(correct / total)
        
        # Model 3: BBPM
        model3 = BBPMAssociativeModel(embedding, d_model, mem_cfg, vocab_size).to(device)
        param_count3 = count_parameters(model3)
        
        # Check parameter count difference
        param_diff_pct = abs(param_count3 - param_count1) / param_count1 * 100
        if param_diff_pct > 5:
            print(f"Warning: Model3 param count differs by {param_diff_pct:.1f}% from Model1")
        
        optimizer3 = optim.Adam(model3.parameters(), lr=lr)
        
        train_losses3 = []
        test_accs3 = []
        
        for epoch in range(num_epochs):
            model3.train()
            epoch_loss = 0
            for i in range(0, len(train_seqs), batch_size):
                batch_seqs = train_seqs[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                optimizer3.zero_grad()
                outputs = model3(batch_seqs)  # [batch, vocab_size]
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer3.step()
                epoch_loss += loss.item()
            
            train_losses3.append(epoch_loss / (len(train_seqs) // batch_size))
            
            model3.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i in range(0, len(test_seqs), batch_size):
                    batch_seqs = test_seqs[i:i+batch_size]
                    batch_targets = test_targets[i:i+batch_size]
                    outputs = model3(batch_seqs)  # [batch, vocab_size]
                    # Accuracy: exact match (top-1)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == batch_targets).sum().item()
                    total += len(batch_targets)
                test_accs3.append(correct / total)
        
        raw_trials.append({
            "seed": seed,
            "windowed_transformer": {
                "param_count": param_count1,
                "train_losses": train_losses1,
                "test_accs": test_accs1,
                "final_test_acc": test_accs1[-1],
            },
            "transformer_external_kv": {
                "param_count": param_count2,
                "train_losses": train_losses2,
                "test_accs": test_accs2,
                "final_test_acc": test_accs2[-1],
            },
            "bbpm": {
                "param_count": param_count3,
                "train_losses": train_losses3,
                "test_accs": test_accs3,
                "final_test_acc": test_accs3[-1],
            },
        })
    
    # Summarize
    summary = {}
    for model_name in ["windowed_transformer", "transformer_external_kv", "bbpm"]:
        final_accs = [t[model_name]["final_test_acc"] for t in raw_trials]
        param_counts = [t[model_name]["param_count"] for t in raw_trials]
        
        acc_stats = mean_ci95(final_accs)
        mean_acc = acc_stats["mean"]
        acc_lo = acc_stats["ci95_low"]
        acc_hi = acc_stats["ci95_high"]
        acc_std = acc_stats["std"]
        mean_params = np.mean(param_counts)
        
        summary[model_name] = {
            "final_test_acc": {
                "mean": mean_acc,
                "ci95_low": acc_lo,
                "ci95_high": acc_hi,
                "std": acc_std,
            },
            "param_count": mean_params,
        }
    
    # Generate figure with 3 panels
    fig = plt.figure(figsize=(16, 5))
    
    epochs = list(range(num_epochs))
    
    # Panel 1: Training loss curves
    ax1 = plt.subplot(1, 3, 1)
    for model_name, label in [
        ("windowed_transformer", "Windowed Transformer"),
        ("transformer_external_kv", "Transformer + External KV"),
        ("bbpm", "BBPM"),
    ]:
        losses = [np.mean([t[model_name]["train_losses"][e] for t in raw_trials]) for e in epochs]
        ax1.plot(epochs, losses, "o-", label=label, linewidth=2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (CrossEntropy)")
    ax1.set_title("Training Loss Curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Test accuracy curves
    ax2 = plt.subplot(1, 3, 2)
    for model_name, label in [
        ("windowed_transformer", "Windowed Transformer"),
        ("transformer_external_kv", "Transformer + External KV"),
        ("bbpm", "BBPM"),
    ]:
        accs = [np.mean([t[model_name]["test_accs"][e] for t in raw_trials]) for e in epochs]
        ax2.plot(epochs, accs, "o-", label=label, linewidth=2)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (Exact Match)")
    ax2.set_title("Test Accuracy Curves")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: Final accuracy bar chart with CI
    ax3 = plt.subplot(1, 3, 3)
    model_names = ["windowed_transformer", "transformer_external_kv", "bbpm"]
    labels = ["Windowed\nTransformer", "Transformer\n+ External KV", "BBPM"]
    final_accs = []
    acc_lows = []
    acc_highs = []
    for model_name in model_names:
        final_accs.append(summary[model_name]["final_test_acc"]["mean"])
        acc_lows.append(summary[model_name]["final_test_acc"]["ci95_low"])
        acc_highs.append(summary[model_name]["final_test_acc"]["ci95_high"])
    
    x_pos = np.arange(len(labels))
    bars = ax3.bar(x_pos, final_accs, yerr=[np.array(final_accs) - np.array(acc_lows), 
                                             np.array(acc_highs) - np.array(final_accs)],
                   capsize=5, alpha=0.7)
    ax3.set_ylabel("Final Test Accuracy")
    ax3.set_title("Final Accuracy (with 95% CI)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add param counts as text
    for i, model_name in enumerate(model_names):
        param_count = summary[model_name]["param_count"]
        ax3.text(i, final_accs[i] + 0.02, f"{int(param_count/1e3)}K params",
                ha="center", va="bottom", fontsize=8)
    
    add_footer(fig, EXP_ID)
    
    # Save outputs
    metrics_path, figure_path = make_output_paths(out_dir, EXP_ID, EXP_SLUG)
    
    config_dict = {
        "batch_size": batch_size,
        "T": T,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_epochs": num_epochs,
        "lr": lr,
        "B": B,
        "L": L,
        "K": K,
        "H": H,
        "device": str(device),
        "dtype": dtype_str,
    }
    
    write_metrics_json(
        metrics_path,
        EXP_ID,
        "End-to-end associative recall. Task: Given sequence of (key, value) pairs as token IDs, "
        "retrieve value token ID for query key. Uses stable keying: hx = mix64(u64(token_id) ^ u64(master_seed)). "
        "Embeddings are trainable to allow models to learn token representations.",
        config_dict,
        seeds,
        raw_trials,
        summary,
    )
    
    save_pdf(fig, figure_path)
    
    return {
        "metrics_path": str(metrics_path),
        "figure_path": str(figure_path),
    }
