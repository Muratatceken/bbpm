---
name: BBPM Theory-Compatible Implementation
overview: Refactor BBPM implementation to use true PRP-based intra-block addressing (Feistel network) instead of hash-based offset mapping, ensuring theory-compatible bijective addressing with no self-collisions.
todos:
  - id: "1"
    content: Implement addressing/block_selector.py with vectorized mix64() (with MASK64 masking) and select_block() functions (NO Python loops over batch, independent seed from PRP)
    status: completed
  - id: "2"
    content: Implement addressing/prp_feistel.py with vectorized Feistel PRP (feistel_prp_vectorized with fixed 6-round loop, derive_round_keys_vectorized, prp_offsets) - operates on exact bit domain n_bits=log2(L) where n_bits is even, NO modulo after PRP, all operations masked with MASK64
    status: completed
  - id: "3"
    content: Implement addressing/bbpm_addressing.py with BBPMAddressing class combining block selection + PRP, enforce K<=L, power-of-two, and even n_bits validation
    status: completed
  - id: "4"
    content: Write test_prp.py with bijection tests (vectorized, optimized for CI - full test for small L, sampled rows for large L), exact bit domain verification, determinism, and CPU/GPU equivalence test (exact equality with MASK64 masking)
    status: completed
  - id: "5"
    content: Write test_addressing.py with offset distinctness (verify PRP guarantees), address range, multi-hash independence tests
    status: completed
  - id: "6"
    content: Write test_memory_ops.py with write/read identity tests using BBPMAddressing, verify no self-collisions
    status: completed
  - id: "7"
    content: Update BBPMMemoryFloat to support block_size parameter with hash_fn override logic (warn if both provided)
    status: completed
  - id: "8"
    content: Add deprecation warnings to BlockHash class
    status: completed
  - id: "9"
    content: Create addressing/__init__.py and update all __init__.py files to export BBPMAddressing
    status: completed
  - id: "10"
    content: Create examples/minimal_demo.py and examples/needle_sanity.py demonstrating PRP-based addressing
    status: completed
  - id: "11"
    content: Update README.md with theory alignment, addressing specification (block selection + intra-block PRP), and constraints
    status: completed
  - id: "12"
    content: Update USAGE_GUIDE.md with BBPMAddressing usage, migration path from BlockHash, and PRP vs hash-based differences
    status: completed
---

