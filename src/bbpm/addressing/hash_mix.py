"""Hash mixing functions for addressing.

This module provides deterministic 64-bit hash mixing functions based on
SplitMix64 algorithm. All functions operate on Python ints and are
platform-independent, ensuring deterministic behavior across runs.
"""


def u64(x: int) -> int:
    """Force integer into unsigned 64-bit domain.

    Args:
        x: Input integer (can be negative or any size)

    Returns:
        Unsigned 64-bit integer (value modulo 2^64)
    """
    return x & 0xFFFFFFFFFFFFFFFF


def mix64(x: int) -> int:
    """64-bit mixing function based on SplitMix64.

    Applies three rounds of XOR-shift and multiplication to thoroughly
    mix the bits of a 64-bit value. This is a stateless mixing function
    that provides excellent avalanche properties.

    Args:
        x: Input value (will be masked to 64 bits)

    Returns:
        Mixed 64-bit unsigned integer

    Example:
        >>> mix64(42)
        12345678901234567890  # Example output (actual value will differ)
    """
    # Mask to 64 bits
    z = u64(x)

    # Round 1: XOR-shift right 30, multiply
    z = u64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)

    # Round 2: XOR-shift right 27, multiply
    z = u64((z ^ (z >> 27)) * 0x94D049BB133111EB)

    # Round 3: Final XOR-shift right 31
    return u64(z ^ (z >> 31))


def derive_seed(base: int, salt: int) -> int:
    """Derive a seed from base and salt using mixing.

    Combines base and salt values using XOR, then applies mixing to
    produce a well-distributed seed value.

    Args:
        base: Base seed value
        salt: Salt value to mix with base

    Returns:
        Mixed seed value (64-bit unsigned)

    Example:
        >>> derive_seed(42, 123)
        9876543210987654321  # Example output
    """
    return mix64(u64(base) ^ u64(salt))


def make_salts(n: int, master_seed: int) -> list[int]:
    """Generate deterministic list of salts from master seed.

    Uses repeated mixing to generate n distinct salts. Each salt is
    derived by mixing the previous state, ensuring deterministic
    generation while maintaining good distribution.

    Args:
        n: Number of salts to generate
        master_seed: Master seed value

    Returns:
        List of n distinct 64-bit unsigned salts

    Example:
        >>> salts = make_salts(3, 42)
        >>> len(salts)
        3
        >>> all(isinstance(s, int) for s in salts)
        True
    """
    salts = []
    state = u64(master_seed)
    for _ in range(n):
        state = mix64(state)
        salts.append(state)
    return salts
