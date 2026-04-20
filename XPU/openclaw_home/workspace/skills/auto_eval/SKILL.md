---
name: auto_eval
description: Automatically evaluate quantized LLM models using vLLM inference engine and lm-evaluation-harness. Supports Intel XPU device with proper quantization format detection and configuration.
metadata:
  openclaw:
    emoji: "📊"
    homepage: https://github.com/EleutherAI/lm-evaluation-harness
    skillKey: auto-eval
    requires:
      bins: ["lm_eval", "vllm"]
      env: []
      config: []
---

# Auto-Eval Skill

Use this skill when users want to evaluate quantized LLM models (especially Auto-Round quantized models) using vLLM and lm-evaluation-harness on Intel XPU.

## Overview

This skill provides a complete workflow for:
- **Detecting quantization format** from model metadata
- **Configuring vLLM** for XPU device with proper settings
- **Running lm-eval** benchmarks (e.g., piqa, hellaswag, mmlu)
- **Handling common errors** and optimizations

**Supported tasks**: piqa, hellaswag, mmlu, arc, gsm8k, and many more via lm-eval

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `model_path` | Path to quantized model (local or HF) | Yes | - |
| `task` | Evaluation task (e.g., piqa, mmlu) | Yes | - |
| `output_path` | Directory for results | No | `./lm_eval_results` |
| `batch_size` | Batch size for evaluation | No | `8` |
| `device` | Device (xpu, cuda, cpu) | No | `xpu` |
| `max_model_len` | Max sequence length | No | `8192` |
| `gpu_memory_utilization` | VRAM fraction | No | `0.8` |

---

## Step 1: Analyze Model and Detect Quantization Format

### Check quantization_config.json

```bash
# For Auto-Round quantized models
cat {model_path}/quantization_config.json
```

**Common quantization formats and vLLM support:**

| Format | packing_format | vLLM Support | Device |
|--------|---------------|---------------|--------|
| auto_gptq | `auto_gptq` | ✓ (symmetric) | XPU, CUDA |
| auto_awq | `auto_awq` | ✓ (asymmetric) | CUDA |
| auto_round | `auto_round:auto_gptq` | ✓ | XPU, CUDA |
| llm_compressor | `auto_round:llm_compressor` | ✓ (MXFP4/NVFP4) | XPU, CUDA |
| GGUF | `gguf` | ✓ (llama.cpp) | CPU |

### Example quantization_config.json for different formats:

**Auto-GPTQ format:**
```json
{
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "quant_method": "auto-gptq",
  "packing_format": "auto_gptq"
}
```

**Auto-Round format:**
```json
{
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}
```

**LLM-Compressor (MXFP4) format:**
```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "config_groups": {...},
  "provider": "auto-round"
}
```

---

## Step 1.5: Check for Shared Workspace (model_info.json)

**IMPORTANT: Before setting up any environment, check if `auto_run` or `auto_quant` has already set up a venv for this model.**

The `auto_run` skill writes a `model_info.json` to the shared workspace directory. If it exists, reuse the venv from it instead of installing from scratch.

The shared workspace directory is typically the `auto_run` output directory for this model:
- e.g., `/storage/lkk/inference/Qwen_Qwen3-0.6B/model_info.json`
- The task prompt may explicitly specify it as `workspace_dir`

```python
import json
from pathlib import Path

workspace_dir = "{workspace_dir}"   # e.g. /storage/lkk/inference/Qwen_Qwen3-0.6B
info_path = Path(workspace_dir) / "model_info.json"

if info_path.exists():
    model_info = json.loads(info_path.read_text())
    venv_path = model_info["venv_path"]    # e.g. /storage/.../venv
    venv_pip  = f"{venv_path}/bin/pip"
    venv_py   = f"{venv_path}/bin/python"
    print(f"✅ Reusing shared venv from auto_run: {venv_path}")
    # → Install lm_eval and vllm into this venv if not already present
    # venv_pip install lm_eval vllm
else:
    print("ℹ️  No model_info.json found, will use system vllm/lm_eval")
    venv_py = "python3"
    venv_pip = "pip3"
```

---

## Step 2: Configure vLLM Model Args

Based on quantization format, configure appropriate vLLM arguments:

### For XPU Device (Intel GPU)

**Recommended settings for XPU:**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
tensor_parallel_size=1,\
max_model_len=8192,\
max_num_batched_tokens=32768,\
max_num_seqs=128,\
gpu_memory_utilization=0.8,\
max_gen_toks=2048,\
enforce_eager=True
```

### Key Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `enforce_eager=True` | Disable CUDA graph for XPU stability | **Required for XPU** |
| `dtype` | Data type | `bfloat16` for XPU |
| `max_model_len` | Max sequence length | 8192 (adjust based on VRAM) |
| `gpu_memory_utilization` | VRAM fraction | 0.8 (leaves headroom) |
| `tensor_parallel_size` | Tensor parallelism | 1 for single XPU |

### Format-Specific Model Args

**For auto_gptq format:**
```python
model_args = "pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,enforce_eager=True,gpu_memory_utilization=0.8"
```

**For auto_round format:**
```python
model_args = "pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,enforce_eager=True,gpu_memory_utilization=0.8"
```

**For llm_compressor (MXFP4/NVFP4) format:**
```python
model_args = "pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,enforce_eager=True,gpu_memory_utilization=0.8"
```

---

## Step 3: Run lm-eval

### Basic Command

```bash
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,max_gen_toks=2048,enforce_eager=True \
    --tasks piqa \
    --batch_size 8 \
    --output_path lm_eval_results \
    --device xpu
```

### Running Multiple Tasks

```bash
# Multiple tasks
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,enforce_eager=True \
    --tasks piqa,hellaswag,mmlu \
    --batch_size 8 \
    --output_path lm_eval_results

# Task group
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,enforce_eager=True \
    --tasks arc_easy,arc_challenge,piqa,hellaswag \
    --batch_size 8
```

### Using lm_eval Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,enforce_eager=True,gpu_memory_utilization=0.8",
    tasks="piqa",
    batch_size="auto",
    device="xpu"
)

print(results["results"])
print(results["versions"])
```

---

## Step 4: Common Tasks

### Popular Benchmarks

| Task | Description | Few-shot |
|------|-------------|----------|
| piqa | Physical interaction QA | 0 |
| hellaswag | commonsense inference | 10 |
| mmlu | multitask language understanding | 5 |
| arc | abstract reasoning | 0 |
| gsm8k | grade school math | 3 |
| winogrande | Winograd schema | 0 |
| boolq | boolean questions | 0 |
| cb | commitment bank | 0 |
| copa | causal reasoning | 0 |

### List Available Tasks

```bash
lm_eval ls tasks
```

---

## Step 5: Troubleshooting

### Common Errors and Solutions

#### 1. vLLM Backend Not Found

**Error:**
```
ValueError: Unknown model: vllm
```

**Solution:**
```bash
# Install lm-eval with vllm support
pip install lm-eval[torch,vllm]

# Or ensure vllm is importable
python -c "import vllm; print(vllm.__version__)"
```

#### 2. XPU Device Not Available

**Error:**
```
AssertionError: XPU device not found
```

**Solution:**
```bash
# Check XPU availability
python -c "import torch; print(torch.xpu.is_available())"

# Set device explicitly
export SYCL_DEVICE_TYPE=GPU
export ONEAPI_VART_DEVICES=0
```

#### 3. Quantization Format Not Supported

**Error:**
```
RuntimeError: Unsupported quantization format
```

**Solution:**
```python
# Check quantization_config.json format
# For unsupported formats, use Transformers instead of vLLM
model_args = "pretrained=./model,device_map=auto,torch_dtype=auto"
```

#### 4. OOM on XPU

**Error:**
```
RuntimeError: XPU out of memory
```

**Solutions:**
```bash
# Reduce max_model_len
--model_args ...max_model_len=2048,gpu_memory_utilization=0.5

# Use enforce_eager=True
--model_args ...enforce_eager=True

# Reduce batch_size
--batch_size 4
```

#### 5. CUDA Graph Not Supported on XPU

**Error:**
```
RuntimeError: CUDA graph is not supported on XPU
```

**Solution:**
```bash
# MUST use enforce_eager=True for XPU
--model_args ...enforce_eager=True
```

#### 6. Import Error: vllm module

**Error:**
```
ModuleNotFoundError: No module named 'vllm'
```

**Solution:**
```bash
# Install vllm with XPU support
pip install vllm --index-url https://download.pytorch.org/whl/xpu
```

---

## Step 6: Evaluation Script Template

### Complete Evaluation Script

```python
#!/usr/bin/env python3
"""
Auto-Eval Script for Quantized Models
"""

import argparse
import json
import os
from pathlib import Path

def detect_quantization_format(model_path: str) -> dict:
    """Detect quantization format from model metadata."""
    quant_config_path = Path(model_path) / "quantization_config.json"
    
    if not quant_config_path.exists():
        return {"format": "bf16", "method": "none"}
    
    with open(quant_config_path) as f:
        config = json.load(f)
    
    # Detect format
    quant_method = config.get("quant_method", "")
    packing_format = config.get("packing_format", "")
    
    if "auto-round" in quant_method:
        if "llm_compressor" in packing_format:
            return {"format": "llm_compressor", "method": "auto-round"}
        return {"format": "auto_round", "method": "auto-round"}
    elif "auto-gptq" in quant_method:
        return {"format": "auto_gptq", "method": "gptq"}
    elif "auto-awq" in quant_method:
        return {"format": "auto_awq", "method": "awq"}
    elif "compressed-tensors" in quant_method:
        return {"format": "compressed_tensors", "method": "auto-round"}
    
    return {"format": "unknown", "method": quant_method}


def build_vllm_args(model_path: str, device: str = "xpu", **kwargs) -> str:
    """Build vLLM model arguments string."""
    
    # Base args
    args = [
        f"pretrained={model_path}",
        "dtype=bfloat16",
        "tensor_parallel_size=1",
    ]
    
    # Add user overrides
    max_model_len = kwargs.get("max_model_len", 8192)
    gpu_mem_util = kwargs.get("gpu_memory_utilization", 0.8)
    batch_tokens = kwargs.get("max_num_batched_tokens", 32768)
    max_seqs = kwargs.get("max_num_seqs", 128)
    max_gen_toks = kwargs.get("max_gen_toks", 2048)
    
    args.extend([
        f"max_model_len={max_model_len}",
        f"max_num_batched_tokens={batch_tokens}",
        f"max_num_seqs={max_seqs}",
        f"gpu_memory_utilization={gpu_mem_util}",
        f"max_gen_toks={max_gen_toks}",
    ])
    
    # XPU requires enforce_eager=True
    if device == "xpu":
        args.append("enforce_eager=True")
    
    return ",".join(args)


def run_evaluation(
    model_path: str,
    tasks: str,
    output_path: str = "./lm_eval_results",
    batch_size: int = 8,
    device: str = "xpu",
    **kwargs
):
    """Run lm-eval evaluation."""
    import lm_eval
    
    # Detect format
    quant_info = detect_quantization_format(model_path)
    print(f"Detected quantization: {quant_info}")
    
    # Build model args
    model_args = build_vllm_args(model_path, device, **kwargs)
    print(f"Model args: {model_args}")
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=str(batch_size),
        device=device,
        output_path=output_path
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, "results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Eval for Quantized Models")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--tasks", required=True, help="Comma-separated tasks")
    parser.add_argument("--output", default="./lm_eval_results", help="Output dir")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="xpu", choices=["xpu", "cuda", "cpu"])
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        tasks=args.tasks,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
```

### Usage Example

```bash
# Run evaluation
python eval_script.py \
    --model /kaokao/quantized \
    --tasks piqa,hellaswag \
    --output ./results \
    --batch_size 8 \
    --device xpu \
    --max_model_len 8192
```

---

## Step 7: Environment Setup

### Install Dependencies

```bash
# For XPU support
pip install torch --index-url https://download.pytorch.org/whl/xpu

# Install vllm with XPU support
pip install vllm --index-url https://download.pytorch.org/whl/xpu

# Install lm-eval with vllm
pip install lm-eval[torch,vllm]

# Verify installations
python -c "import torch; print('XPU:', torch.xpu.is_available())"
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import lm_eval; print('lm-eval:', lm_eval.__version__)"
```

---

## Step 8: Quick Reference

### Complete Example for Auto-Round W4A16 Model

```bash
# Model path
MODEL_PATH="/kaokao/quantized"

# Run piqa evaluation
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,max_gen_toks=2048,enforce_eager=True" \
    --tasks piqa \
    --batch_size 8 \
    --output_path ./lm_eval_results \
    --device xpu
```

### Run Multiple Benchmarks

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,enforce_eager=True" \
    --tasks piqa,hellaswag,mmlu,arc_easy \
    --batch_size 8 \
    --output_path ./results
```

### Using Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=/kaokao/quantized,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,enforce_eager=True,gpu_memory_utilization=0.8",
    tasks="piqa",
    batch_size="auto",
    device="xpu"
)
print(results["results"]["piqa"])
```

---

## Environment Variables for XPU

```bash
# Intel XPU environment
export SYCL_DEVICE_TYPE=GPU
export ONEAPI_VART_DEVICES=0
export ZE_AFFINITY_MASK=0

# Disable CUDA to force XPU
export CUDA_VISIBLE_DEVICES=""
```

---

## Notes

- **XPU requires `enforce_eiver=True`** - CUDA graphs not supported
- **Batch size** - Use `auto` or small values (4-8) for stability
- **Quantization formats** - Auto-Round exports to `auto_round` format with `packing_format` metadata
- **VRAM** - Leave ~20% headroom with `gpu_memory_utilization=0.8`
- **Tasks** - Common: piqa, hellaswag, mmlu, arc, gsm8k