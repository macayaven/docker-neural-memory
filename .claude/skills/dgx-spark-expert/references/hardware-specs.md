# DGX Spark Hardware Specifications

## GB10 Grace Blackwell Superchip

| Component | Specification |
|-----------|---------------|
| Architecture | NVIDIA Grace Blackwell |
| GPU | Blackwell Architecture |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| CUDA Cores | Blackwell Generation |
| Tensor Cores | 5th Generation |
| RT Cores | 4th Generation |
| Tensor Performance | 1 PFLOP (FP4 with sparsity) |

## Memory Architecture (UMA)

- **Total System Memory**: 128GB LPDDR5x coherent unified memory
- **Memory Interface**: 256-bit
- **Memory Bandwidth**: 273 GB/s
- **Key Concept**: CPU and GPU share the same memory pool (no discrete VRAM)

### UMA Implications
- Models up to 200B parameters can run locally
- No explicit memory transfers between CPU/GPU needed
- `nvidia-smi` may report "Memory-Usage: Not Supported" (expected)
- Use `free -h` to check actual memory availability
- Buffer cache may need manual flushing: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

## Storage
- **Capacity**: 4TB NVMe M.2
- **Feature**: Self-encryption support

## Connectivity

| Port | Specification |
|------|---------------|
| USB | 4× USB Type-C |
| Ethernet | 1× RJ-45 (10 GbE) |
| High-Speed NIC | ConnectX-7 @ 200 Gbps |
| WiFi | WiFi 7 |
| Bluetooth | BT 5.4 |
| Display | 1× HDMI 2.1a |
| NVENC/NVDEC | 1× each |

## Power & Physical

| Spec | Value |
|------|-------|
| Power Supply | 240W |
| GB10 TDP | 140W |
| Dimensions | 150mm × 150mm × 50.5mm |
| Weight | 1.2 kg |

## Spark Stacking (Two-Node Cluster)
- Connect two DGX Spark units via QSFP/CX7 cable
- Combined: Up to 405B parameter models
- Uses NCCL v2.28.3 for GPU-to-GPU communication
- MPI for inter-process CPU communication
- CX-7 ports support Ethernet configuration only

## Architecture Notes (ARM64)
- System runs on ARM64 architecture (not x86)
- Requires ARM64-compatible binaries and containers
- NGC CLI must use ARM64 Linux version
- Some x86-only software won't work without emulation
