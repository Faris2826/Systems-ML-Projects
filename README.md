# Meta SWE Intern Bundle – Faris Ali Khan

## Projects
| Folder | Tech | One-liner |
|---|---|---|
| Diffusion-MeshGen | SDS loss | Text → textured mesh in 8 s |
| Vulkan-ForwardPlus | C++ | Clustered lights, 42 % GPU save |
| TensorRT-LLM | Python | 2.3× throughput vs vanilla PyTorch |
| ML-SuperRes-VR | PyTorch | 4× upsample &lt; 11 ms on Quest 3 |


### Vulkan Forward+ Renderer
- High-performance rendering engine (C++, Vulkan)
- Supports 100+ dynamic lights at 60 FPS
- Includes lock-free job system for parallel work

### TensorRT LLM Inference
- Optimised LLM inference on GPU (CUDA, TensorRT)
- Reduced binary size and improved throughput
- Achieved ~128 tokens/sec on RTX 3060

### Stable-DreamFusion (3D Generation)
- Text-to-3D mesh generation pipeline
- Improved stability and fixed CUDA compatibility issues

### AIMET VR Super-Resolution
- Implemented INT8-optimised ESRGAN model for VR applications
- Achieved ~1.8× performance improvement while maintaining 90 FPS
- Focused on real-time inference and optimisation for VR workloads
