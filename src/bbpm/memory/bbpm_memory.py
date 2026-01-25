"""Core BBPM memory implementation.

This module provides the canonical Block-Based Permutation Memory implementation
that implements the IBbpmMemory protocol with deterministic addressing,
configurable normalization, and precision-safe accumulation.
"""

import torch
import torch.nn as nn

from bbpm.addressing.block_address import AddressConfig, BlockAddress
from bbpm.memory.interfaces import IBbpmMemory, MemoryConfig


class BBPMMemory(nn.Module):
    """Canonical Block-Based Permutation Memory implementation.

    Implements IBbpmMemory protocol with deterministic addressing,
    configurable normalization, and precision-safe accumulation.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        """Initialize BBPM memory.

        Args:
            cfg: Memory configuration
        """
        super().__init__()
        self.cfg = cfg

        # Compute total memory size D = num_blocks * block_size
        D = cfg.num_blocks * cfg.block_size

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map[cfg.dtype]

        # Initialize memory tensor [D, d]
        self.register_buffer(
            "memory",
            torch.zeros(D, cfg.key_dim, dtype=self.dtype, device=cfg.device),
        )

        # Initialize counts tensor if count_normalized mode
        if cfg.read_mode == "count_normalized":
            self.register_buffer(
                "counts",
                torch.zeros(D, 1, dtype=torch.float32, device=cfg.device),
            )
        else:
            # Don't register None buffer - use attribute instead
            self.counts = None

        # Initialize BlockAddress for deterministic addressing
        addr_cfg = AddressConfig(
            num_blocks=cfg.num_blocks,
            block_size=cfg.block_size,
            K=cfg.K,
            H=cfg.H,
            master_seed=cfg.master_seed,
        )
        self.addresser = BlockAddress(addr_cfg)

        # Epsilon for normalization stability
        self.eps = 1e-8

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize value according to cfg.normalize_values.

        Args:
            v: Value tensor of shape [d] or [1, d]

        Returns:
            Normalized value tensor of shape [d]
        """
        # Ensure v is [d] shape
        if v.dim() == 2 and v.shape[0] == 1:
            v = v.squeeze(0)

        if self.cfg.normalize_values == "none":
            return v
        elif self.cfg.normalize_values == "l2":
            norm = torch.norm(v, p=2) + self.eps
            return v / norm
        elif self.cfg.normalize_values == "rms":
            rms = torch.sqrt(torch.mean(v**2)) + self.eps
            return v / rms
        else:
            raise ValueError(
                f"Unknown normalize_values: {self.cfg.normalize_values}"
            )

    def write(self, hx: int, v: torch.Tensor) -> None:
        """Write value to memory at addresses derived from hashed key.

        Input tensor v will be automatically moved to memory device if needed.

        Args:
            hx: Hashed item key (uint64)
            v: Value tensor of shape [d] or [1, d]

        Raises:
            ValueError: If input shape is invalid
            TypeError: If input type is invalid
        """
        # Validate input shape
        if v.dim() not in (1, 2):
            raise ValueError(f"v must be [d] or [1,d], got shape {v.shape}")
        if v.dim() == 2:
            if v.shape[0] != 1:
                raise ValueError(f"v must be [d] or [1,d], got shape {v.shape}")
            v = v.squeeze(0)
        if v.shape[0] != self.cfg.key_dim:
            raise ValueError(
                f"v shape[0] ({v.shape[0]}) must match key_dim ({self.cfg.key_dim})"
            )

        # Ensure v is on correct device
        if v.device != self.memory.device:
            v = v.to(self.memory.device)

        # Normalize value
        v_norm = self._normalize(v)

        # Compute addresses as tensor directly (no Python list intermediate)
        device = torch.device(self.cfg.device)
        addr_tensor = self.addresser.addresses_tensor(hx, device)

        # Expand v_norm to [H*K, d] for index_add
        v_expanded = v_norm.unsqueeze(0).expand(len(addr_tensor), -1)

        # Implement dtype accumulation policy
        if self.dtype == torch.float32:
            # Native accumulation for float32
            self.memory.index_add_(0, addr_tensor, v_expanded)
        elif self.dtype == torch.bfloat16:
            # bfloat16 accumulation: only fast_inexact mode allowed
            if self.cfg.accumulate == "fast_inexact":
                # Only convert values being written, not full memory tensor
                # This ensures O(H*K*d) complexity, not O(D*d)
                self.memory.index_add_(0, addr_tensor, v_expanded.to(self.dtype))
            else:
                # This should never happen due to config validation, but check anyway
                raise ValueError(
                    "bfloat16 requires accumulate='fast_inexact'. "
                    "Safe accumulation with full-tensor casts is not supported "
                    "(O(D*d) complexity violation)."
                )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        # Update counts if count_normalized mode
        if self.counts is not None:
            ones = torch.ones(
                len(addr_tensor), 1, dtype=torch.float32, device=device
            )
            self.counts.index_add_(0, addr_tensor, ones)

    def read(self, hx: int) -> torch.Tensor:
        """Read value from memory using hashed key.

        Args:
            hx: Hashed item key (uint64)

        Returns:
            Retrieved value tensor of shape [d] with dtype from cfg.output_dtype
        """
        # Compute addresses as tensor directly
        device = torch.device(self.cfg.device)
        addr_tensor = self.addresser.addresses_tensor(hx, device)

        # Gather values at addresses [H*K, d]
        gathered = self.memory[addr_tensor]

        # Apply count normalization if needed
        if self.cfg.read_mode == "count_normalized" and self.counts is not None:
            counts_at_addr = self.counts[addr_tensor]  # [H*K, 1]
            # Normalize by counts: memory / max(count, 1)
            counts_safe = torch.clamp(counts_at_addr, min=1.0)
            gathered = gathered / counts_safe

        # Mean over H*K addresses to get [d]
        result = gathered.mean(dim=0)

        # Convert to output dtype (default float32)
        output_dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        output_dtype = output_dtype_map[self.cfg.output_dtype]
        result = result.to(output_dtype)

        return result

    def write_batch(self, hx_tensor: "torch.LongTensor", values: "torch.Tensor") -> None:
        """Write a batch of values to memory.
        
        Args:
            hx_tensor: Tensor of shape [T] containing uint64 hashed keys (as int64)
            values: Tensor of shape [T, d] containing values to write
        """
        T = hx_tensor.shape[0]
        for i in range(T):
            hx = int(hx_tensor[i].item())
            v = values[i]
            self.write(hx, v)

    def read_batch(self, hx_tensor: "torch.LongTensor") -> "torch.Tensor":
        """Read a batch of values from memory.
        
        Args:
            hx_tensor: Tensor of shape [T] containing uint64 hashed keys (as int64)
            
        Returns:
            Tensor of shape [T, d] containing retrieved values
        """
        T = hx_tensor.shape[0]
        results = []
        for i in range(T):
            hx = int(hx_tensor[i].item())
            r = self.read(hx)
            results.append(r)
        return torch.stack(results, dim=0)

    def reset(self) -> None:
        """Reset memory to initial state (clear all contents)."""
        self.memory.zero_()
        if self.counts is not None:
            self.counts.zero_()

    def stats(self) -> dict:
        """Get memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        D = self.cfg.num_blocks * self.cfg.block_size

        stats_dict = {
            "D": D,
            "dtype": self.cfg.dtype,
            "device": self.cfg.device,
            "read_mode": self.cfg.read_mode,
            "normalize_values": self.cfg.normalize_values,
        }

        # Add count statistics if available
        if self.counts is not None:
            total_writes = self.counts.sum().item()
            occupied_slots = (self.counts > 0).sum().item()
            stats_dict["total_writes"] = total_writes
            stats_dict["occupied_slots"] = occupied_slots
            stats_dict["empty_slots"] = D - occupied_slots
            if occupied_slots > 0:
                stats_dict["mean_writes_per_slot"] = total_writes / occupied_slots
            else:
                stats_dict["mean_writes_per_slot"] = 0.0

        return stats_dict
