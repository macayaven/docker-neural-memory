# DGX Spark Known Issues & Troubleshooting

## Official Known Issues

### Power Adapter
**Issue**: Use only the supplied 240W power adapter.
**Symptoms**: Reduced performance, boot failures, unexpected shutdowns.
**Solution**: Always use the included power adapter.

### nvidia-smi Memory Display
**Issue**: `nvidia-smi` reports "Memory-Usage: Not Supported"
**Cause**: Expected behavior on UMA (Unified Memory Architecture) systems.
**Workaround**: Use `free -h` to check system memory instead.

### Memory Reporting with UMA
**Issue**: Memory tools may misreport available GPU memory.
**Details**: DGX Spark shares 128GB between CPU and GPU dynamically. Tools expecting discrete VRAM may show incorrect values.
**For developers**: Don't rely solely on `cudaMemGetInfo()`. Consider DRAM reclamation via SWAP.

## Common Troubleshooting

### Display Issues
**Symptom**: No display output on first boot.
**Solutions**:
1. Try HDMI instead of USB-C/DisplayPort
2. Attach peripherals BEFORE connecting power
3. Some displays need specific timing on boot

### Memory Pressure / OOM Errors
**Symptom**: "CUDA out of memory" or "OutOfMemoryError"
**Solutions**:
```bash
# Flush buffer cache
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Check actual memory
free -h

# Monitor memory usage
watch -n 1 free -h
```

### Driver Communication Failure
**Symptom**: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"
**Solutions**:
```bash
# Check driver status
sudo systemctl status nvidia-persistenced

# Reload driver
sudo modprobe -r nvidia && sudo modprobe nvidia

# Full system reboot if needed
sudo reboot
```

### Network Connectivity Issues
**Symptoms**: WiFi, Bluetooth, or Ethernet not working.
**Solutions**:
1. Check `ip link` for interface status
2. For Ethernet: Look for NO-CARRIER status
3. Ensure netplan configuration is correct
4. Reboot after network config changes

### Container/Docker Issues
**Check runtime**:
```bash
docker info | grep -i runtime
# Should show nvidia runtime

# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### Software Not Working (ARM64)
**Cause**: DGX Spark uses ARM64 architecture.
**Solutions**:
- Use ARM64/aarch64 versions of software
- Check NGC for ARM64 containers
- Some x86-only tools won't work

## Multi-Node Troubleshooting

### Node 2 Not Visible
**Symptoms**: Second Spark not joining cluster, Ray cluster issues, Docker Swarm join fails.
**Solutions**:
1. Verify CX-7 cable connection
2. Check netplan config on both nodes
3. Verify SSH keys are exchanged
4. Test with: `ssh sparkuser@<node2-ip> hostname`

### NCCL Communication Failures
**Solutions**:
```bash
# Verify NCCL installation
python3 -c "import torch; print(torch.cuda.nccl.version())"

# Check network interfaces
ibstat  # For InfiniBand/RoCE
ip addr show  # For Ethernet

# Run NCCL tests
nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 1
```

## Performance Issues

### Slow Inference
**Checks**:
1. Model quantization (use FP4/INT8 for speed)
2. Batch size optimization
3. Memory pressure (flush cache)
4. Use optimized backends (TRT-LLM, vLLM)

### Model Loading Slow
**Solutions**:
1. Use safetensors format
2. Pre-download to NVMe storage
3. Use quantized model versions
4. Check storage I/O: `iostat -x 1`

## Diagnostics Commands

```bash
# System overview
nvidia-smi
free -h
df -h
uname -a

# GPU info
nvidia-smi -q

# CPU info
lscpu

# Storage
nvme list
sudo smartctl -a /dev/nvme0

# Network
ip addr
ethtool <interface>

# Docker
docker info
docker ps -a

# Services
systemctl status nvidia-persistenced
systemctl status docker

# Logs
journalctl -u nvidia-persistenced
dmesg | tail -50
```

## Support Resources

- **Forums**: https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10
- **User Guide**: https://docs.nvidia.com/dgx/dgx-spark/
- **Support Portal**: https://nvidia.custhelp.com/
- **Live Chat**: http://nvidia.custhelp.com/app/chat/chat_launch/
