"""Experiment 05: End-to-end associative recall."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bbpm.memory.interfaces import MemoryConfig
from bbpm.memory.bbpm_memory import BBPMMemory
from common import (
    make_output_paths,
    seed_loop,
    ensure_device,
    write_metrics_json,
)
from plotting import save_pdf, add_footer
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
        default=1000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
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
    d_model: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate canonical associative recall task batch.
    
    Task: Given sequence of (key, value) pairs, retrieve value for a query key.
    Sequence format: [key1, val1, key2, val2, ..., query_key, <pad>]
    Target: value corresponding to query_key (exact match retrieval).
    
    Args:
        batch_size: Batch size
        T: Sequence length (number of pairs + query)
        vocab_size: Vocabulary size
        d_model: Model dimension
        
    Returns:
        (sequences, targets, target_indices)
        sequences: [batch_size, T, d_model] token embeddings
        targets: [batch_size, d_model] target value embeddings
        target_indices: [batch_size] indices of target in sequence
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random embeddings for vocabulary
    vocab_embeddings = torch.randn(vocab_size, d_model, device=device)
    vocab_embeddings = torch.nn.functional.normalize(vocab_embeddings, p=2, dim=1)
    
    sequences = []
    targets = []
    target_indices = []
    
    for _ in range(batch_size):
        # Generate pairs: (key, value) pairs
        num_pairs = (T - 1) // 2
        pairs = []
        used_keys = set()
        
        for _ in range(num_pairs):
            key_id = random.randint(0, vocab_size - 1)
            val_id = random.randint(0, vocab_size - 1)
            pairs.append((key_id, val_id))
            used_keys.add(key_id)
        
        # Query key: must appear in pairs
        query_key_id = random.choice(list(used_keys))
        
        # Find target value
        target_val_id = None
        for k, v in pairs:
            if k == query_key_id:
                target_val_id = v
                break
        
        # Build sequence: [key1, val1, key2, val2, ..., query_key, <pad>]
        seq_embeddings = []
        target_idx = None
        for i, (k, v) in enumerate(pairs):
            seq_embeddings.append(vocab_embeddings[k])
            seq_embeddings.append(vocab_embeddings[v])
            if k == query_key_id and target_idx is None:
                target_idx = len(seq_embeddings) - 1  # Value position
        
        # Add query
        seq_embeddings.append(vocab_embeddings[query_key_id])
        if target_idx is None:
            # Find in pairs
            for k, v in pairs:
                if k == query_key_id:
                    for j, (k2, v2) in enumerate(pairs):
                        if k2 == query_key_id:
                            target_idx = j * 2 + 1
                            break
                    break
        
        # Pad to T
        while len(seq_embeddings) < T:
            seq_embeddings.append(torch.zeros(d_model, device=device))
        
        sequences.append(torch.stack(seq_embeddings[:T]))
        targets.append(vocab_embeddings[target_val_id])
        target_indices.append(target_idx)
    
    return torch.stack(sequences), torch.stack(targets), torch.tensor(target_indices, device=device)


class WindowedTransformer(nn.Module):
    """Windowed transformer baseline."""
    
    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4),
            num_layers=2,
        )
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T, d_model]"""
        # Use last window_size tokens
        x_window = x[:, -self.window_size:, :]
        # [window_size, batch, d_model]
        x_window = x_window.transpose(0, 1)
        out = self.transformer(x_window)
        # Use last token
        return self.output_proj(out[-1].transpose(0, 1))


class TransformerWithExternalKV(nn.Module):
    """Transformer with external learned KV memory."""
    
    def __init__(self, d_model: int, num_heads: int, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4),
            num_layers=2,
        )
        self.attention_to_memory = nn.MultiheadAttention(d_model, num_heads)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, T, d_model]"""
        # Process sequence
        x_seq = x.transpose(0, 1)  # [T, batch, d_model]
        x_encoded = self.transformer(x_seq)
        query = x_encoded[-1:]  # Last token [1, batch, d_model]
        
        # Attend to external memory
        memory = self.memory.unsqueeze(1).expand(-1, query.size(1), -1)  # [M, batch, d_model]
        attn_out, _ = self.attention_to_memory(query, memory, memory)
        return self.output_proj(attn_out.squeeze(0))


class BBPMAssociativeModel(nn.Module):
    """BBPM-based associative recall model."""
    
    def __init__(self, d_model: int, mem_cfg: MemoryConfig):
        super().__init__()
        self.mem = BBPMMemory(mem_cfg)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, hx_list: list[int]) -> torch.Tensor:
        """x: [batch, T, d_model], hx_list: list of keys for last token"""
        # Write sequence to memory
        self.mem.reset()
        batch_size = x.shape[0]
        T = x.shape[1]
        
        for t in range(T - 1):  # All but last (query)
            for b in range(batch_size):
                # Use token embedding hash as key
                token_emb = x[b, t]
                # Simple hash: sum of embedding (deterministic)
                hx = int(torch.sum(token_emb * 1000).item()) % (2**64)
                self.mem.write(hx, token_emb)
        
        # Retrieve using query
        retrieved = []
        for b in range(batch_size):
            query_emb = x[b, -1]
            hx = int(torch.sum(query_emb * 1000).item()) % (2**64)
            r = self.mem.read(hx)
            retrieved.append(r)
        
        retrieved_tensor = torch.stack(retrieved)
        return self.output_proj(retrieved_tensor)


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
    lr = 1e-3
    
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
        
        # Generate train/test data
        train_seqs, train_targets, _ = generate_batch(
            batch_size * 10, T, vocab_size, d_model, seed, device
        )
        test_seqs, test_targets, _ = generate_batch(
            batch_size * 5, T, vocab_size, d_model, seed + 1000, device
        )
        
        # Model 1: Windowed Transformer
        window_size = 16
        model1 = WindowedTransformer(d_model, num_heads, window_size).to(device)
        param_count1 = count_parameters(model1)
        
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
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
                outputs = model1(batch_seqs)
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
                    outputs = model1(batch_seqs)
                    # Accuracy: cosine similarity > 0.9
                    cosines = torch.nn.functional.cosine_similarity(outputs, batch_targets, dim=1)
                    correct += (cosines > 0.9).sum().item()
                    total += len(batch_targets)
                test_accs1.append(correct / total)
        
        # Model 2: Transformer + External KV
        memory_size = 1000
        model2 = TransformerWithExternalKV(d_model, num_heads, memory_size).to(device)
        param_count2 = count_parameters(model2)
        
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
                outputs = model2(batch_seqs)
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
                    outputs = model2(batch_seqs)
                    cosines = torch.nn.functional.cosine_similarity(outputs, batch_targets, dim=1)
                    correct += (cosines > 0.9).sum().item()
                    total += len(batch_targets)
                test_accs2.append(correct / total)
        
        # Model 3: BBPM
        model3 = BBPMAssociativeModel(d_model, mem_cfg).to(device)
        param_count3 = count_parameters(model3)
        
        optimizer3 = optim.Adam(model3.parameters(), lr=lr)
        
        train_losses3 = []
        test_accs3 = []
        
        for epoch in range(num_epochs):
            model3.train()
            epoch_loss = 0
            for i in range(0, len(train_seqs), batch_size):
                batch_seqs = train_seqs[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                # Generate hx_list for this batch
                batch_hx_list = []
                for b in range(batch_seqs.shape[0]):
                    seq_hx = []
                    for t in range(batch_seqs.shape[1]):
                        token_emb = batch_seqs[b, t]
                        hx = int(torch.sum(token_emb * 1000).item()) % (2**64)
                        seq_hx.append(hx)
                    batch_hx_list.append(seq_hx)
                
                optimizer3.zero_grad()
                outputs = model3(batch_seqs, batch_hx_list[0])  # Use first sequence's keys
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
                    batch_hx_list = []
                    for b in range(batch_seqs.shape[0]):
                        seq_hx = []
                        for t in range(batch_seqs.shape[1]):
                            token_emb = batch_seqs[b, t]
                            hx = int(torch.sum(token_emb * 1000).item()) % (2**64)
                            seq_hx.append(hx)
                        batch_hx_list.append(seq_hx)
                    outputs = model3(batch_seqs, batch_hx_list[0])
                    cosines = torch.nn.functional.cosine_similarity(outputs, batch_targets, dim=1)
                    correct += (cosines > 0.9).sum().item()
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
        
        mean_acc, acc_lo, acc_hi, acc_std = mean_ci95(final_accs)
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
    
    # Generate figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Training curves
    epochs = list(range(num_epochs))
    for model_name, label in [
        ("windowed_transformer", "Windowed Transformer"),
        ("transformer_external_kv", "Transformer + External KV"),
        ("bbpm", "BBPM"),
    ]:
        losses = [np.mean([t[model_name]["train_losses"][e] for t in raw_trials]) for e in epochs]
        ax1.plot(epochs, losses, "o-", label=label, linewidth=2)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss (MSE)")
    ax1.set_title("Training Curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Test accuracy curves
    for model_name, label in [
        ("windowed_transformer", "Windowed Transformer"),
        ("transformer_external_kv", "Transformer + External KV"),
        ("bbpm", "BBPM"),
    ]:
        accs = [np.mean([t[model_name]["test_accs"][e] for t in raw_trials]) for e in epochs]
        ax2.plot(epochs, accs, "o-", label=label, linewidth=2)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (cosine > 0.9)")
    ax2.set_title("Test Accuracy Curves")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
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
        f"{EXP_ID}_{EXP_SLUG}",
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
