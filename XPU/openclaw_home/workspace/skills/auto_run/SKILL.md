---
name: auto_run
description: Run inference tests on Large Language Models (LLMs) and Vision-Language Models (VLMs) using HuggingFace Transformers. Tests if models can run on CUDA/XPU devices.
metadata:
  openclaw:
    emoji: "🚀"
    homepage: https://huggingface.co/docs/transformers
    skillKey: auto-run
    requires:
      bins: []
      env: []
      config: []
---

# Auto-Run Model Inference Skill

Use this skill when you need to test if a model can run on CUDA or XPU devices. This skill provides comprehensive guidance for inference testing, including error handling, troubleshooting, and model-specific optimizations.

## Overview

This skill helps you:
1. Test NLP text-generation models on CUDA/XPU
2. Test Vision-Language models (VLM) on CUDA/XPU
3. Handle dependency issues and model-specific bugs
4. Generate minimal reproduction scripts

**Supported model types:**
- Causal LM (text generation): LLaMA, Qwen, Mistral, GPT, etc.
- Vision-Language Models: LLaVA, Qwen-VL, Penguin-VL, etc.

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `model_path` | HuggingFace model ID or local path | Yes | - |
| `transformers_path` | Local transformers source path | No | - |
| `model_type` | Type: `nlp` or `vl` | No | Auto-detect |
| `device` | Device: `cuda`, `cuda:0`, `xpu`, `cpu` | No | `cuda:0` |
| `dtype` | Data type: `auto`, `bfloat16`, `float16`, `float32` | No | `auto` |
| `prompt` | Test prompt for generation | No | Generic prompt |
| `trust_remote_code` | Allow executing remote code | No | `True` |

---

## Step 1: Analyze Model from HuggingFace

**CRITICAL: Always fetch model information before testing.**

### Fetch Model Card and Config

```bash
# README (model card) - contains usage instructions
curl -L https://huggingface.co/{model_id}/resolve/main/README.md -o /tmp/{model_id}_README.md

# config.json - architecture details
curl -L https://huggingface.co/{model_id}/resolve/main/config.json -o /tmp/{model_id}_config.json

# tokenizer_config.json
curl -L https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json -o /tmp/{model_id}_tokenizer.json
```

### What to Look For

1. **Model type**: Check `config.json` → `model_type`
   - Common types: `llama`, `qwen`, `mistral`, `gemma`, `falcon`
   - **VLM Types**: If model_type contains `vl`, `vision`, or uses `vision_encoder` → it's a VLM

2. **Recommended: Check transformers_version** - Always read this field from config.json first!
   - If `transformers_version` exists (e.g., `"4.51.3"`), **install the SPECIFIC version first**
   - **Strategy**:
     1. **First**: Try the exact specified version (e.g., `pip install transformers==4.51.3`)
     2. **If fails**: Analyze the error, try fixing with patches or minor adjustments to that version
     3. **If still fails**: Then try nearby versions (4.51.x, 4.52.x, etc.)
     4. **If all fail**: Use your judgment to try other versions
   - Only if `transformers_version` is NOT found, use the fallback:
     - **First try**: `pip install transformers` (latest version)
     - **If fails due to version**: Try older versions (4.51, 4.40, etc.)

3. **Inference instructions**: Search README for:
   - "inference", "quickstart", "usage", "example"
   - Special requirements for loading

4. **Architecture specifics**:
   - Attention implementation (`_attn_implementation`)
   - Torch dtype requirements
   - Trust remote code requirements

---

## Step 2: Set Up Environment (MANDATORY)

**IMPORTANT: Always use an isolated virtual environment. Never run on system Python.**

### Step 2.0 — Reuse Existing Environment (CHECK FIRST!)

**Before creating anything, check if a venv already exists in the output directory:**

```bash
if [ -f {output_dir}/venv/bin/activate ]; then
    echo "EXISTING VENV FOUND — reusing it"
    source {output_dir}/venv/bin/activate
    python -c "import torch; print('torch:', torch.__version__)" && echo "VENV OK"
fi
```

**If the venv exists and torch imports correctly → skip Step 2.1–2.4 entirely.** Just activate it and proceed to Step 3.

Only create a new venv if:
- No venv exists at `{output_dir}/venv/`
- The existing venv is broken (torch import fails)
- The existing venv has the wrong device backend

Similarly, **do NOT re-download model weights** if they already exist in `$HF_HOME`:
```bash
# Check if model is already cached
ls $HF_HOME/hub/models--{org}--{model_name}/snapshots/ 2>/dev/null && echo "MODEL CACHED"
```

### Step 2.1 — Check What's Already in the System

```bash
# Check if system torch exists and which backends it supports
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
try:
    print('XPU available:', torch.xpu.is_available())
    print('XPU count:', torch.xpu.device_count())
except AttributeError:
    print('XPU: not supported by this torch build')
" 2>/dev/null || echo "torch: NOT installed in system python"

pip3 list 2>/dev/null | grep -iE "torch|triton|flash|intel"
```

### Step 2.2 — Decide How to Set Up the Venv

The target device comes from the task prompt (e.g. `Device: xpu:0` or `Device: cuda:0`).
Use this decision table — **do NOT hardcode any assumption**:

| System torch | Target device | Action |
|---|---|---|
| Installed, matches target | cuda or xpu | `--system-site-packages`, skip torch install |
| Installed CUDA, target is **xpu** | xpu | Create plain venv, install XPU torch |
| Installed XPU, target is **cuda** | cuda | Create plain venv, install CUDA torch |
| NOT installed | cuda | Create plain venv, install CUDA torch |
| NOT installed | xpu | Create plain venv, install XPU torch |

### Step 2.3 — Create Venv and Install torch

```bash
mkdir -p {output_dir}/logs {output_dir}/patches

# If system torch matches target device → reuse it
python3 -m venv {output_dir}/venv --system-site-packages
# Otherwise (system torch absent or wrong backend) → plain venv
python3 -m venv {output_dir}/venv

source {output_dir}/venv/bin/activate
pip install -U pip setuptools wheel
```

**Install torch matching the target device (only when system torch is absent or wrong):**

```bash
# For CUDA target:
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

# For XPU target:
# ⚠️ plain `pip install torch` always installs the CUDA build — NEVER use it for XPU
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/xpu
```

**Verify torch matches the target device after install:**

```bash
# For CUDA:
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# For XPU:
python -c "import torch; print('torch:', torch.__version__); print('XPU:', torch.xpu.is_available()); print('XPU count:', torch.xpu.device_count())"
```

### Step 2.4 — Adjustable Packages (Safe to Install/Upgrade)

- `transformers` (version as needed)
- `numpy` (downgrade to `<2` if conflicts occur)
- `pillow`, `decord`, `einops`, `opencv-python`
- `tokenizers`, `huggingface-hub`

**Never reinstall torch/flash_attn/pytorch-triton if system already has them for the correct backend.**

### Install Dependencies

**Step 1: Verify system packages are accessible**

```bash
# After activating venv with --system-site-packages
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')" 2>/dev/null || echo "flash_attn: not available"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Step 2: Install transformers (local source or PyPI)**

**RECOMMENDED: First check the model's config.json for `transformers_version`!**

```bash
# Step 2a: Check required transformers version (if field exists)
curl -L https://huggingface.co/{model_id}/resolve/main/config.json | grep transformers_version

# Step 2b: Install version - if transformers_version found, install SPECIFIC version
# If found (e.g., 4.51.3):
pip install "transformers==4.51.3"

# If that fails, try nearby versions:
pip install "transformers>=4.51.3,<4.52"  # Try 4.51.x
pip install "transformers>=4.52,<4.53"    # Try 4.52.x

# If not found - start with latest, then adjust based on errors:
pip install transformers  # Try latest first
# If fails with version errors, try: pip install "transformers>=4.51" or "transformers>=4.40"

# If user provides local transformers path, ensure it's compatible
pip install -e /storage/lkk/transformers

# Otherwise install a safe default from PyPI
pip install transformers
```

**Step 3: Install additional dependencies as needed**

```bash
# For VLM models (minimal set for testing)
pip install pillow

# Optional: decord, einops, opencv-python (if processor requires)
pip install decord einops opencv-python

# For specific tokenizers
pip install sentencepiece protobuf tiktoken
```

**⚠️ NEVER do these:**
- ❌ `pip install torch` (will overwrite system torch)
- ❌ `pip install flash-attn` (will overwrite system flash_attn)
- ❌ `pip install torch==<version>` (changes system version)

If you can't install a specific version → rely on `--system-site-packages`

### Common Issue: NumPy Version Conflict

When installing VLM dependencies, numpy may be upgraded to 2.x which is incompatible with some packages:

```bash
# Fix: downgrade numpy (if needed)
pip install "numpy<2"
```

### Common Issue: Transformers Version Mismatch (Recommended Check!)

**First check the model's config.json for `transformers_version` field - if found, use it!**

```bash
# Step 1: Fetch config.json
curl -L https://huggingface.co/{model_id}/resolve/main/config.json

# Step 2: Check for transformers_version field
# Example output:
# "transformers_version": "4.51.3"
```

If `transformers_version` exists:
```bash
# Install the SPECIFIC version first (not >=)
pip install "transformers==4.51.3"

# If fails, analyze error and try nearby versions:
pip install "transformers>=4.51.3,<4.52"  # Try 4.51.x
pip install "transformers>=4.52,<4.53"    # Try 4.52.x

# If still fails, try patching or other versions based on error analysis
```

If it doesn't exist (flexible - not all models have this field):
- Qwen3/VLM models: try `transformers>=4.40`
- Newer models (2025+): try `transformers>=4.51`
- Or just use latest: `pip install transformers`

**Version Compatibility Troubleshooting:**

| Error | Solution |
|-------|----------|
| `offload_state_dict` not found | transformers too old - check config.json for required version first, then try nearby versions |
| `device_map` / `meta device` errors | transformers too new - try the exact specified version from config.json |
| `is_flash_attn_greater_or_equal` not found | transformers too old - try specific version from config or nearby versions |

```bash
# If config.json specifies transformers_version (e.g., 4.51.3):
pip install "transformers==4.51.3"  # Try exact version first

# If that fails, try nearby minor versions:
pip install "transformers>=4.51.3,<4.52"  # Try 4.51.x
pip install "transformers>=4.52,<4.53"    # Try 4.52.x

# If no specific version in config, try:
pip install "transformers>=4.40,<5.0"   # Try 4.x
pip install "transformers>=4.51,<4.52" # Try specific 4.51.x
pip install "transformers>=4.57,<5.0"  # Try latest 4.x

# Or use local source
pip install -e /path/to/transformers
```

### Auto-Detect and Install Correct Transformers Version

Use this script to automatically detect and install the correct transformers version:

```python
#!/usr/bin/env python3
"""
Auto-detect and install correct transformers version based on model config
"""

import json
import subprocess
import sys

def get_transformers_version_from_config(model_id: str) -> str:
    """Fetch transformers_version from model config.json"""
    import urllib.request
    
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            config = json.loads(response.read().decode())
            return config.get("transformers_version")
    except Exception as e:
        print(f"Warning: Could not fetch config: {e}")
        return None

def install_transformers(required_version: str = None):
    """Install transformers with specific version"""
    if required_version:
        cmd = f"pip install 'transformers>={required_version}'"
    else:
        cmd = "pip install transformers"
    
    print(f"Installing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error installing: {result.stderr}")
        return False
    
    print("Successfully installed transformers")
    return True

# Example usage:
if __name__ == "__main__":
    model_id = "{model_id}"  # Replace with actual HuggingFace model ID
    
    # Step 1: Get required version from config
    required_version = get_transformers_version_from_config(model_id)
    
    if required_version:
        print(f"Model requires transformers >= {required_version}")
    else:
        print("No specific transformers_version found, will try latest version")
        required_version = None  # Will use latest
    
    # Step 2: Install
    install_transformers(required_version)
```

### Environment Variables for GPU Testing

**IMPORTANT**: Set CUDA_VISIBLE_DEVICES BEFORE importing torch in your test script:

```python
# MUST be set before any torch import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import torch
# Now torch.cuda.is_available() will return True
```

---

## Step 3: Run Inference Test

### For NLP Models (Text Generation)

```python
#!/usr/bin/env python3
"""
Auto-Run Inference Test - NLP Model
Model: {model_path}
Device: {device}
"""

import os
import sys
import logging

# CRITICAL: Set device BEFORE importing torch
gpu_device = "{gpu_device}"  # e.g., "0" or "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# Set up logging
log_file = os.path.join("{output_dir}", "logs", "inference.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
model_name_or_path = "{model_path}"
device = "cuda:{gpu_device}" if "{device}".startswith("cuda") else "{device}"
dtype = "{dtype}"   # "auto", "bfloat16", "float16"
prompt = "{prompt}"

logger.info(f"Testing model: {model_name_or_path}")
logger.info(f"Device: {device}")
logger.info(f"Dtype: {dtype}")
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Verify CUDA available
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load model
logger.info("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    logger.info("Model loaded successfully!")
    
    # Move to GPU if CUDA available
    if torch.cuda.is_available():
        model = model.to(device)
        logger.info(f"Model device: {model.device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Try with device_map="auto"
    logger.info("Retrying with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)

# Handle missing pad token
if tokenizer.pad_token is None:
    logger.warning("pad_token not found, using eos_token as pad_token")
    tokenizer.pad_token = tokenizer.eos_token

# Run inference
logger.info(f"Running inference with prompt: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt")

# Move to device
if device != "cpu":
    device_model = model.device
    inputs = {k: v.to(device_model) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
else:
    device_model = "cpu"

logger.info(f"Input device: {device_model}")
logger.info(f"Input IDs: {inputs['input_ids'].shape}")

# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

# Decode output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
logger.info(f"Output: {output_text}")

# Verify output
if len(output_text) > len(prompt):
    logger.info("✅ Inference test PASSED!")
else:
    logger.warning("⚠️ Output seems empty or too short")

logger.info("Test complete!")
```

### For Vision-Language Models (VLM)

```python
#!/usr/bin/env python3
"""
Auto-Run Inference Test - VLM Model
Model: {model_path}
Device: {device}
"""

import os
import sys
import logging

# CRITICAL: Set device BEFORE importing torch
gpu_device = "{gpu_device}"  # e.g., "0" or "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# Set up logging
log_file = os.path.join("{output_dir}", "logs", "inference.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Configuration
model_name_or_path = "{model_path}"
device = "cuda:{gpu_device}" if "{device}".startswith("cuda") else "{device}"
dtype = "{dtype}"

logger.info(f"Testing VLM model: {model_name_or_path}")
logger.info(f"Device: {device}")
logger.info(f"Dtype: {dtype}")
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Verify CUDA available
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load model (without device_map to avoid meta device issues)
logger.info("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype="bfloat16" if dtype == "auto" else dtype,
        attn_implementation="eager",  # Avoid flash_attn issues
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Move model to GPU
if torch.cuda.is_available():
    logger.info("Moving model to GPU...")
    model = model.to(device)
    logger.info(f"Model device: {model.device}")

# Load processor
logger.info("Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    logger.info("Processor loaded!")
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.info("Install: pip install pillow decord einops opencv-python")
    sys.exit(1)

# Prepare inputs with dummy image
logger.info("Preparing inputs...")
from PIL import Image
import numpy as np

dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "Describe this image."

try:
    inputs = processor(
        conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": prompt}
                ],
            },
        ],
        return_tensors="pt",
    )
except Exception as e:
    logger.warning(f"Conversation format failed: {e}, trying simpler format...")
    inputs = processor(
        text=[prompt],
        images=[dummy_image],
        return_tensors="pt",
    )

# Move inputs to device
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
          for k, v in inputs.items()}

logger.info(f"Input keys: {inputs.keys()}")
if "pixel_values" in inputs:
    logger.info(f"Pixel values shape: {inputs['pixel_values'].shape}")

# Generate
logger.info("Running inference...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

# Decode
response = processor.decode(output_ids[0], skip_special_tokens=True)
logger.info(f"Output: {response}")

if len(response) > 0:
    logger.info("✅ VLM Inference test PASSED!")
else:
    logger.warning("⚠️ Output seems empty")

logger.info("Test complete!")
```

---

## Step 4: Execute and Log

```bash
# Activate virtual environment
source {output_dir}/venv/bin/activate

# Run inference test with CUDA_VISIBLE_DEVICES set
# Note: Must be set BEFORE running the script
export CUDA_VISIBLE_DEVICES={gpu_device}
cd {output_dir}
python inference_script.py 2>&1 | tee logs/inference.log
```

**Alternative: Set in script (RECOMMENDED)**

The script template already sets `os.environ["CUDA_VISIBLE_DEVICES"]` at the very top, before importing torch. This is the most reliable way.

---

## Step 5: Error Handling

When inference fails, follow this workflow:

```
ERROR → Analyze → Search → Try Solutions → Verify → Document
```

### Diagnostic Steps

1. **Check the log file:**
```bash
tail -100 {output_dir}/logs/inference.log
```

2. **Identify error type:**
   - Import errors → Check dependencies
   - Model loading errors → Check model card / apply patches
   - CUDA/XPU errors → Check device availability
   - Tokenization errors → Check tokenizer config

3. **Common errors and solutions:**

| Error Type | Solution |
|------------|----------|
| CUDA out of memory | Reduce batch size, use float16 |
| No module named 'transformers' | Install transformers |
| Trust remote code error | Add `trust_remote_code=True` |
| Tokenizer not found | Check model ID or local path |
| Device not found | Check CUDA/XPU availability |
| dtype not supported | Use `torch_dtype="auto"` |
| pad_token not found | Set `tokenizer.pad_token = tokenizer.eos_token` |
| Vision encoder error | For VLM: ensure correct processor |
| Image processing error | Install pillow, check image format |

4. **After fixing, re-run:**
```bash
source {output_dir}/venv/bin/activate
python inference_script.py 2>&1 | tee -a logs/inference.log
```

5. **Save error and solution:**
```bash
cat >> {output_dir}/SOLUTIONS.md << 'EOF'
### Error: [Description]
- **Solution**: [What worked]
- **Commands**: [Commands used]
EOF
```

---

## Step 6: Generate Summary (RECOMMENDED)

After inference test completes (success or failure), generate a `summary.md` to document the entire process. This helps with debugging, reproducibility, and tracking issues.

### Summary Template

```python
#!/usr/bin/env python3
"""
Generate inference test summary
Run this after inference test completes (success or failure)
"""

import json
import os
from datetime import datetime
from pathlib import Path

def generate_summary(
    output_dir: str,
    model_path: str,
    model_type: str,
    device: str,
    dtype: str,
    start_time: float,
    errors: list = None,
    solutions: list = None,
    notes: str = None
):
    """Generate a comprehensive summary markdown file."""
    
    import time
    end_time = time.time()
    duration = end_time - start_time
    
    # Collect output files
    output_path = Path(output_dir)
    files_info = []
    if output_path.exists():
        for f in sorted(output_path.rglob("*")):
            if f.is_file() and not f.name.endswith(('.pyc', '.pyo', '__pycache__')):
                size = f.stat().st_size
                size_str = f"{size/1024/1024:.2f} MB" if size > 1024*1024 else f"{size/1024:.2f} KB"
                files_info.append(f"  - {f.relative_to(output_path)} ({size_str})")
    
    # Build summary markdown
    summary = f"""# Inference Test Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Model Information

| Field | Value |
|-------|-------|
| Model Path | `{model_path}` |
| Model Type | `{model_type}` |
| Device | `{device}` |
| Dtype | `{dtype}` |

## Timing

| Phase | Duration |
|-------|----------|
| Total | {duration:.2f} seconds ({duration/60:.2f} minutes) |

## Output Files

```
{chr(10).join(files_info) if files_info else "  (no files found)"}
```

## Errors Encountered

{chr(10).join(f"- {err}" for err in (errors or ["(none)"]))}

## Solutions Applied

{chr(10).join(f"- {sol}" for sol in (solutions or ["(none)"]))}

## Additional Notes

{notes or "(none)"}

## Environment

```bash
# Python version
python3 --version

# Key packages
pip show torch transformers
```

## Reproduce Command

```bash
# Recreate this inference test
# Set CUDA device BEFORE importing torch
export CUDA_VISIBLE_DEVICES={device.split(':')[-1] if ':' in device else device}
python inference_script.py
```
"""
    
    # Write summary
    summary_path = Path(output_dir) / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    
    print(f"Summary written to: {summary_path}")
    return summary_path

# Usage example:
if __name__ == "__main__":
    import time
    start_time = time.time()  # Set this at the beginning of inference test
    
    # Your inference code here...
    
    # Generate summary at the end
    generate_summary(
        output_dir="{output_dir}",
        model_path="{model_path}",
        model_type="{model_type}",
        device="{device}",
        dtype="{dtype}",
        start_time=start_time,
        errors=["ImportError: No module named 'transformers'", "Fixed by installing transformers>=5.0"],
        solutions=["pip install transformers>=5.0"],
        notes="VLM inference test completed successfully"
    )
```

### Integration with Inference Script

Add summary generation to your inference script:

```python
#!/usr/bin/env python3
import time
import json
from pathlib import Path

# Track start time
start_time = time.time()

# Track errors and solutions
errors = []
solutions = []

try:
    # Your inference code here
    model = AutoModelForCausalLM.from_pretrained(...)
    # ... run inference ...
    
except Exception as e:
    errors.append(str(e))
    
    # Try to recover
    try:
        # Attempted solution 1
        solutions.append("Installed transformers>=5.0")
    except:
        pass
    
    finally:
        # Always generate summary even if inference failed
        generate_summary(
            output_dir=output_dir,
            model_path=model_path,
            model_type=model_type,
            device=device,
            dtype=dtype,
            start_time=start_time,
            errors=errors,
            solutions=solutions,
            notes="Inference test failed, see errors above"
        )
        raise

# Success path - generate summary
generate_summary(
    output_dir=output_dir,
    model_path=model_path,
    model_type=model_type,
    device=device,
    dtype=dtype,
    start_time=start_time,
    notes="Inference test completed successfully"
)
```

### Summary Output Example

The generated `summary.md` will look like:

```markdown
# Inference Test Summary

Generated: 2026-03-20 01:07 UTC

## Model Information

| Field | Value |
|-------|-------|
| Model Path | `tencent/Penguin-VL-2B` |
| Model Type | `vl` |
| Device | `cuda:0` |
| Dtype | `bfloat16` |

## Timing

| Phase | Duration |
|-------|----------|
| Total | 45.32 seconds (0.76 minutes) |

## Output Files

```
- inference_script.py (5.67 KB)
- logs/inference.log (12.34 KB)
- summary.md (2.15 KB)
```

## Errors Encountered

- (none)

## Solutions Applied

- (none)

## Additional Notes

- VLM inference test completed successfully

## Environment

```bash
# Python version
Python 3.10.12

# Key packages
torch: 2.8.0a0+34c6371d24.nv25.8
transformers: 4.40.0
```

## Reproduce Command

```bash
# Recreate this inference test
export CUDA_VISIBLE_DEVICES=0
python inference_script.py
```
```

---

## Quick Reference Card

| Need | Solution |
|------|----------|
| Test NLP model | Use `AutoModelForCausalLM` + `AutoTokenizer` |
| Test VLM model | Use `AutoModelForCausalLM` + `AutoProcessor` |
| Use specific GPU | Set `CUDA_VISIBLE_DEVICES=0` in script **before** `import torch` |
| Reuse system CUDA packages | `python3 -m venv venv --system-site-packages` |
| Adjust transformers version | First check config.json for `transformers_version`, then install that specific version |
| Use local transformers | `pip install -e /storage/lkk/transformers` |
| Use XPU | Set `device="xpu"` |
| Lower memory | Use `torch_dtype="float16"` |
| Fix pad_token | `tokenizer.pad_token = tokenizer.eos_token` |
| Trust remote code | Add `trust_remote_code=True` |
| VLM dependencies | `pip install pillow decord einops opencv-python` |
| Fix numpy conflict | `pip install "numpy<2"` |

### ⚠️ NEVER MODIFY (CUDA-coupled)
- ❌ torch version
- ❌ flash_attn version
- ❌ pytorch-triton version

### ✅ CAN ADJUST (not CUDA-coupled)
- ✅ transformers version
- ✅ numpy version
- ✅ Other Python packages

---

## Notes

- **CUDA-coupled packages (DO NOT MODIFY)**: torch, flash_attn, pytorch-triton - these are tied to CUDA drivers
- **Adjustable packages**: transformers, numpy, and other non-CUDA packages can be freely modified
- **USE `--system-site-packages`**: Preferred way to reuse system CUDA packages without modification
- **Patches are allowed**: If you encounter issues that require patching, you can apply patches (see Known Issues section)
- **VRAM Requirements**: ~2-4GB for 1B models, ~8-16GB for 8B models
- **Time**: ~10-30 seconds for basic inference test
- **Test prompts**: Use simple, generic prompts for initial testing
- **ALWAYS use virtual environment**: Never pollute system Python
- **ALWAYS log to file**: Use `tee` or logging to capture output
- **Check model card**: For model-specific requirements
- **CUDA_VISIBLE_DEVICES**: Must be set BEFORE `import torch` in your script

---

## Known Issues and Solutions

### 0. Environment Detection (CRITICAL - Always Check First)

**Problem**: Installing torch/flash_attn from scratch may result in incompatible versions or miss CUDA support.

**Solution - ALWAYS check system environment first:**

```bash
# Check what packages already exist in system Python
python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')" 2>/dev/null
pip3 list | grep -iE "torch|triton|flash"

# When creating venv, copy/reuse system packages:
pip install --no-deps torch==<system_version>
pip install --no-deps pytorch-triton==<system_version>
pip install --no-deps flash-attn==<system_version>
```

### 1. CUDA Out of Memory

**Solution:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 instead of bfloat16
    device_map="auto",          # Let device_map handle memory
    low_cpu_mem_usage=True,     # Reduce CPU memory during loading
)
```

### 2. Trust Remote Code Error

**Solution:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
)
```

### 3. pad_token Not Found

**Solution:**
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 4. VLM Image Processing Error

**Solution:**
```python
# Use PIL to create dummy image for testing
from PIL import Image
import numpy as np
dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
```

### 5. XPU Device — Complete Setup Guide

When `device` is `xpu` or `xpu:N`, follow these rules **without exception**.

#### 5.1 Environment Variables

```python
# MUST be the very first lines before any other import
import os
os.environ["ZE_AFFINITY_MASK"] = "0"   # replace 0 with actual device_index
```

> ⚠️ Do NOT set `CUDA_VISIBLE_DEVICES` on an XPU machine — it has no effect and causes confusion.

#### 5.2 Verify XPU Availability

```bash
python3 -c "import torch; print('XPU available:', torch.xpu.is_available()); print('XPU count:', torch.xpu.device_count())"
```

If XPU is not available, check that `intel_extension_for_pytorch` is installed:
```bash
pip install intel_extension_for_pytorch
```

#### 5.3 Correct Inference Script for XPU

```python
import os
os.environ["ZE_AFFINITY_MASK"] = "0"   # MUST be first, before any import

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Verify XPU
assert torch.xpu.is_available(), "XPU not available!"
device = "xpu:0"

# Load model directly to XPU
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map=device,           # use "balanced" if model doesn't fit on one card
    trust_remote_code=True,
)

# Assert model is actually on XPU
device_str = str(next(model.parameters()).device)
assert device_str.startswith("xpu"), f"Model not on XPU! Got: {device_str}"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

#### 5.4 Forbidden Patterns on XPU

| ❌ Forbidden | ✅ Correct |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | `ZE_AFFINITY_MASK=0` |
| `torch.cuda.is_available()` | `torch.xpu.is_available()` |
| `device_map="auto"` | `device_map="xpu:0"` |
| `device_map="cpu"` | `device_map="xpu:0"` |
| `.to("cpu")` | `.to("xpu:0")` |
| `.cuda()` | `.to("xpu:0")` |
| CPU fallback | raise error if XPU fails |

If the model is too large for one card, use `device_map="balanced"` — **never fall back to CPU**.

### 6. VLM Flash Attention Error

**Error:** `NameError: name 'flash_attn_varlen_func' is not defined`

**Root Cause:** For VLM models (like Penguin-VL, LLaVA, Qwen-VL), the **Vision Encoder** often hardcodes flash_attn usage, which is NOT controlled by `attn_implementation` parameter. This is different from the LLM backbone.

**Solution 1: Use SDPA/Eager Attention for LLM (May Not Work for VLM)**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Only affects LLM, not Vision Encoder
)
```

**Solution 2: Check and Fix torch/flash_attn Compatibility (RECOMMENDED)**
```bash
# Check versions
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "from flash_attn import flash_attn_varlen_func; print('flash_attn: OK')" 2>/dev/null

# If torch version mismatch, install compatible versions:
# Option A: Use torch with pre-built flash_attn wheels
pip install torch==2.8.0+cu124

# Option B: Install flash_attn from source (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Option C: Use a different CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Solution 3: Use vLLM or SGLang (RECOMMENDED for Production)**
```python
# vLLM handles flash_attn internally
from vllm import LLM, SamplingParams
llm = LLM(model="{model_id}", tensor_parallel_size=1)
```

**⚠️ Known Issue: Patching Vision Encoder is Complex**
Manually patching the vision encoder to use SDPA is complex because:
- The attention pattern differs (flash_attn uses variable length sequences)
- SDPA requires different input shapes
- Patch often produces incorrect results

**Solution 4: If You Must Patch (Not Recommended)**
```bash
# Save patch to output directory
PATCH_FILE="{output_dir}/patches/vision_encoder_sdpa.patch"

# Only attempt this if you understand the model architecture deeply
# Even with patch, results may be incorrect
```

### 8. VLM Model with transformers_version Field

**Error:** Model fails to load due to transformers version mismatch

**Root Cause:** Some models (especially newer VLM models) specify `transformers_version` in config.json. Using incompatible versions causes:
- `offload_state_dict` error (transformers too old)
- `device_map` / `meta device` errors (transformers too new)
- Other unexpected loading failures

**Note:** Not all models have this field - if not present, use default strategy.

**Solution:**

```bash
# Step 1: Check if model has transformers_version in config.json
curl -L https://huggingface.co/{model_id}/resolve/main/config.json | grep transformers_version

# Step 2: If found, install that SPECIFIC version first
pip install "transformers==X.Y.Z"  # Replace with actual version

# If fails, try nearby minor versions:
pip install "transformers>=X.Y.Z,<X+1.0"  # Try X.Y.x versions
pip install "transformers>=X+1.0,<X+2.0" # Try next minor version
```

### 9. VLM Image Processing Errors

**Error:** `ValueError: Unsupported image path type`

**Solution:** Pass PIL Image directly instead of file path:
```python
from PIL import Image
import numpy as np

# Create test image
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# Use conversation format with PIL Image
inputs = processor(
    conversation=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},  # Pass PIL Image directly
                {"type": "text", "text": "Describe this image."}
            ],
        },
    ],
    return_tensors="pt",
)
```

---

## Quick Checklist

Before testing:
- [ ] Analyze model (config.json, README.md)
- [ ] Identify model type (NLP or VLM)
- [ ] **Check system environment** (torch, flash_attn versions)
- [ ] Create virtual environment **with `--system-site-packages`**
- [ ] Install transformers (adjust version if needed for model compatibility)
- [ ] Install VLM dependencies (pillow, decord, einops, opencv-python)
- [ ] Fix numpy version if needed (`pip install "numpy<2"`)
- [ ] **NEVER reinstall torch/flash_attn** (CUDA-coupled)

During testing:
- [ ] Verify CUDA_VISIBLE_DEVICES is set in script before import torch
- [ ] Run inference script
- [ ] Check log for errors
- [ ] Fix issues: adjust transformers version, apply patches if needed
- [ ] Document solutions

After testing:
- [ ] Verify output is generated
- [ ] Check output is not empty
- [ ] Save final log to SOLUTIONS.md
