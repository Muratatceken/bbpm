"""Pseudo-random permutation (Feistel PRP) implementation.

This module provides a keyed, deterministic permutation over power-of-two
domains using Feistel networks. The implementation uses mix64-based round
functions for strong cryptographic mixing.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bbpm.addressing.hash_mix import mix64, u64

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class FeistelPRP:
    """Feistel network-based pseudo-random permutation.

    Implements a keyed PRP over domain [0, 2^nbits) using Feistel rounds.
    Domain size must be a power of two (L = 2^nbits).

    Attributes:
        nbits: Number of bits (m), domain size is 2**nbits
        rounds: Number of Feistel rounds (>= 6 recommended for security)
        master_key: Master key (uint64) for round constant generation
    """

    nbits: int  # m, domain size is 2**nbits
    rounds: int  # >= 6 recommended
    master_key: int  # uint64
    round_consts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.nbits <= 0:
            raise ValueError("nbits must be positive")
        if self.rounds < 6:
            raise ValueError("rounds should be >= 6 for security")
        if not (0 <= self.master_key < 2**64):
            raise ValueError("master_key must be uint64")
        
        # Precompute round constants once
        round_consts = []
        state = u64(self.master_key)
        for _ in range(self.rounds):
            state = mix64(state)
            round_consts.append(state)
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(self, "round_consts", tuple(round_consts))

    def _split(self, x: int) -> tuple[int, int]:
        """Split x into left and right halves.

        If nbits is odd, right half gets the extra bit.

        Args:
            x: Input value in [0, 2**nbits)

        Returns:
            (left, right) tuple where left has nbits//2 bits,
            right has (nbits+1)//2 bits
        """
        left_bits = self.nbits // 2
        right_bits = (self.nbits + 1) // 2

        right_mask = (1 << right_bits) - 1
        right = x & right_mask
        left = x >> right_bits

        return left, right

    def _combine(self, left: int, right: int) -> int:
        """Combine left and right halves back into full value.

        Args:
            left: Left half (nbits//2 bits)
            right: Right half ((nbits+1)//2 bits)

        Returns:
            Combined value in [0, 2**nbits)
        """
        right_bits = (self.nbits + 1) // 2
        return (left << right_bits) | right

    def _get_round_constant(self, round_idx: int) -> int:
        """Get precomputed round constant for given round.

        Round constants are precomputed in __post_init__ from master_key.

        Args:
            round_idx: Round index (0-based)

        Returns:
            Round constant (uint64)
        """
        return self.round_consts[round_idx]

    def _round_function(self, r: int, round_idx: int, key: int) -> int:
        """Feistel round function F.

        F(r, round_idx, key) = lower_half_bits(mix64(r ^ key ^ round_constant ^ master_key))

        Args:
            r: Right half input
            round_idx: Current round index (0-based)
            key: Per-item key (uint64)

        Returns:
            Round function output (lower half bits)
        """
        # Get round constant (derived deterministically from master_key)
        round_const = self._get_round_constant(round_idx)

        # Combine all inputs: r ^ key ^ round_constant ^ master_key
        combined = u64(r) ^ u64(key) ^ u64(round_const) ^ u64(self.master_key)

        # Apply mixing
        mixed = mix64(combined)

        # Extract lower half bits
        half_bits = (self.nbits + 1) // 2  # Right half size
        return mixed & ((1 << half_bits) - 1)

    def permute(self, x: int, key: int) -> int:
        """Apply PRP permutation.

        Returns PRP_k(x) in [0, 2**nbits). key is per-item seed (uint64).
        Deterministic, bijective for any fixed key.

        Args:
            x: Input value in [0, 2**nbits)
            key: Per-item key (uint64)

        Returns:
            Permuted value in [0, 2**nbits)

        Example:
            >>> prp = FeistelPRP(nbits=8, rounds=6, master_key=42)
            >>> y = prp.permute(100, key=123)
            >>> x_recovered = prp.invert(y, key=123)
            >>> x_recovered == 100
            True
        """
        # Validate inputs
        if not (0 <= x < (1 << self.nbits)):
            raise ValueError(f"x must be in [0, 2**{self.nbits}), got {x}")
        if not (0 <= key < 2**64):
            raise ValueError("key must be uint64")

        # Mask x to domain
        domain_mask = (1 << self.nbits) - 1
        x = x & domain_mask

        # Split into halves
        left, right = self._split(x)

        # Apply Feistel rounds
        for round_idx in range(self.rounds):
            # F(right, round_idx, key)
            f_out = self._round_function(right, round_idx, key)

            # XOR with left
            new_right = left ^ f_out

            # Swap for next round (left becomes right, right becomes new_right)
            left = right
            right = new_right

            # Mask to half sizes
            left_bits = self.nbits // 2
            right_bits = (self.nbits + 1) // 2
            left = left & ((1 << left_bits) - 1)
            right = right & ((1 << right_bits) - 1)

        # Final swap (after all rounds)
        left, right = right, left

        # Combine and mask to domain
        result = self._combine(left, right)
        return result & domain_mask

    def _logical_right_shift(self, x: "torch.LongTensor", shift: int) -> "torch.LongTensor":
        """Perform logical right shift on int64 tensor (simulating uint64).

        For values >= 2^63 (negative in int64), arithmetic shift sign-extends.
        This function ensures logical shift (zero-fill) by masking after shift.

        Args:
            x: Input tensor (int64, but treated as uint64)
            shift: Shift amount (must be < 64)

        Returns:
            Logically right-shifted tensor
        """
        import torch

        if shift >= 64:
            # Shift by 64+ is undefined, return zeros
            return torch.zeros_like(x)
        
        # For logical right shift: keep lower (64 - shift) bits, zero-fill upper shift bits
        # Arithmetic shift on negative fills upper bits with 1s, so we mask them out
        # Mask for lower (64 - shift) bits: (1 << (64 - shift)) - 1
        mask_bits = 64 - shift
        # Create mask as tensor on same device as x (device-safe)
        mask = torch.tensor((1 << mask_bits) - 1, dtype=torch.long, device=x.device)
        
        # Perform arithmetic shift, then mask to get logical shift
        return (x >> shift) & mask

    def _mix64_tensor(self, x: "torch.LongTensor") -> "torch.LongTensor":
        """Vectorized 64-bit mixing function based on SplitMix64.

        Applies three rounds of XOR-shift and multiplication to thoroughly
        mix the bits of 64-bit values. Vectorized version for torch tensors.
        Ensures uint64 logical shift semantics using _logical_right_shift.

        Args:
            x: Input tensor of 64-bit values

        Returns:
            Mixed 64-bit unsigned integer tensor
        """
        import torch

        # U64_MASK: all 64 bits set (0xFFFFFFFFFFFFFFFF)
        U64_MASK = -1
        
        # Mask input to uint64 range
        z = x & U64_MASK

        # Round 1: XOR-shift right 30, multiply
        z_shifted = self._logical_right_shift(z, 30)
        z = ((z ^ z_shifted) * 0xBF58476D1CE4E5B9) & U64_MASK

        # Round 2: XOR-shift right 27, multiply
        z_shifted = self._logical_right_shift(z, 27)
        z = ((z ^ z_shifted) * 0x94D049BB133111EB) & U64_MASK

        # Round 3: Final XOR-shift right 31
        z_shifted = self._logical_right_shift(z, 31)
        return (z ^ z_shifted) & U64_MASK

    def _split_tensor(self, x: "torch.LongTensor") -> tuple["torch.LongTensor", "torch.LongTensor"]:
        """Split tensor x into left and right halves.

        If nbits is odd, right half gets the extra bit.

        Args:
            x: Input tensor of values in [0, 2**nbits)

        Returns:
            (left, right) tuple of tensors
        """
        import torch

        left_bits = self.nbits // 2
        right_bits = (self.nbits + 1) // 2

        right_mask = (1 << right_bits) - 1
        right = x & right_mask
        left = x >> right_bits

        return left, right

    def _combine_tensor(self, left: "torch.LongTensor", right: "torch.LongTensor") -> "torch.LongTensor":
        """Combine left and right halves back into full value.

        Args:
            left: Left half tensor (nbits//2 bits)
            right: Right half tensor ((nbits+1)//2 bits)

        Returns:
            Combined tensor in [0, 2**nbits)
        """
        import torch

        right_bits = (self.nbits + 1) // 2
        return (left << right_bits) | right

    def _round_function_tensor(
        self, r: "torch.LongTensor", round_idx: int, key: int
    ) -> "torch.LongTensor":
        """Vectorized Feistel round function F.

        F(r, round_idx, key) = lower_half_bits(mix64(r ^ key ^ round_constant ^ master_key))

        Args:
            r: Right half input tensor
            round_idx: Current round index (0-based)
            key: Per-item key (uint64)

        Returns:
            Round function output tensor (lower half bits)
        """
        import torch

        # Get round constant (derived deterministically from master_key)
        round_const = self._get_round_constant(round_idx)

        # Combine all inputs: r ^ key ^ round_constant ^ master_key
        # For uint64 values, we need to handle them carefully with torch.long (int64)
        # Approach: convert Python ints to proper int64 representation, then XOR
        device = r.device
        
        # Mask Python ints to uint64 range first
        key_masked = key & 0xFFFFFFFFFFFFFFFF
        round_const_masked = round_const & 0xFFFFFFFFFFFFFFFF
        master_key_masked = self.master_key & 0xFFFFFFFFFFFFFFFF
        
        # Convert to int64: if value >= 2^63, interpret as negative in two's complement
        # This preserves the bit pattern for XOR operations
        def to_int64(u64_val):
            if u64_val < 2**63:
                return u64_val
            else:
                return u64_val - 2**64
        
        # Create scalar tensors that broadcast correctly
        key_t = torch.full((), to_int64(key_masked), dtype=torch.long, device=device)
        round_const_t = torch.full((), to_int64(round_const_masked), dtype=torch.long, device=device)
        master_key_t = torch.full((), to_int64(master_key_masked), dtype=torch.long, device=device)
        
        # XOR operations (broadcasting handles scalar ^ tensor)
        combined = r ^ key_t ^ round_const_t ^ master_key_t
        
        # Mask result to uint64 range: use -1 (all bits set) for int64
        # This is equivalent to 0xFFFFFFFFFFFFFFFF in unsigned interpretation
        combined = combined & (-1)

        # Apply mixing
        mixed = self._mix64_tensor(combined)

        # Extract lower half bits
        half_bits = (self.nbits + 1) // 2  # Right half size
        half_mask = (1 << half_bits) - 1
        return mixed & half_mask

    def permute_tensor(
        self, x: "torch.LongTensor", key: int
    ) -> "torch.LongTensor":
        """Vectorized PRP permutation on tensor.

        Applies PRP permutation to each element of input tensor in parallel.
        Each element should match permute(x[i], key) for scalar version.

        Args:
            x: Input tensor of values in [0, 2**nbits)
            key: Per-item key (uint64)

        Returns:
            Permuted tensor of same shape as x
        """
        import torch

        # Validate inputs
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be torch.LongTensor, got {type(x)}")
        if x.dtype != torch.long:
            raise TypeError(f"x must be torch.long dtype, got {x.dtype}")
        if not (0 <= key < 2**64):
            raise ValueError("key must be uint64")

        # Mask x to domain
        domain_mask = (1 << self.nbits) - 1
        x = x & domain_mask

        # Split into halves
        left, right = self._split_tensor(x)

        # Apply Feistel rounds
        for round_idx in range(self.rounds):
            # F(right, round_idx, key)
            f_out = self._round_function_tensor(right, round_idx, key)

            # XOR with left
            new_right = left ^ f_out

            # Swap for next round (left becomes right, right becomes new_right)
            left = right
            right = new_right

            # Mask to half sizes
            left_bits = self.nbits // 2
            right_bits = (self.nbits + 1) // 2
            left_mask = (1 << left_bits) - 1
            right_mask = (1 << right_bits) - 1
            left = left & left_mask
            right = right & right_mask

        # Final swap (after all rounds)
        left, right = right, left

        # Combine and mask to domain
        result = self._combine_tensor(left, right)
        return result & domain_mask

    def invert(self, y: int, key: int) -> int:
        """Inverse permutation for validation/tests.

        Exactly reverses permute(). For any x, key:
            invert(permute(x, key), key) == x

        Args:
            y: Permuted value in [0, 2**nbits)
            key: Per-item key (uint64)

        Returns:
            Original value in [0, 2**nbits)

        Example:
            >>> prp = FeistelPRP(nbits=8, rounds=6, master_key=42)
            >>> y = prp.permute(100, key=123)
            >>> prp.invert(y, key=123)
            100
        """
        # Validate inputs
        if not (0 <= y < (1 << self.nbits)):
            raise ValueError(f"y must be in [0, 2**{self.nbits}), got {y}")
        if not (0 <= key < 2**64):
            raise ValueError("key must be uint64")

        # Mask y to domain
        domain_mask = (1 << self.nbits) - 1
        y = y & domain_mask

        # Split into halves
        left, right = self._split(y)

        # Reverse the final swap from permute()
        left, right = right, left

        # Apply Feistel rounds in reverse order
        for round_idx in range(self.rounds - 1, -1, -1):
            # Standard Feistel inverse: L_i = R_{i+1} ^ F(L_{i+1}), R_i = L_{i+1}
            # We have L_{i+1} = left, R_{i+1} = right
            # Need to compute: L_i = right ^ F(left), R_i = left
            f_out = self._round_function(left, round_idx, key)
            new_left = right ^ f_out
            new_right = left

            # Update for next iteration
            left = new_left
            right = new_right

            # Mask to half sizes
            left_bits = self.nbits // 2
            right_bits = (self.nbits + 1) // 2
            left = left & ((1 << left_bits) - 1)
            right = right & ((1 << right_bits) - 1)

        # Combine and mask to domain
        result = self._combine(left, right)
        return result & domain_mask
