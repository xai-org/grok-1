# 🛡️ Grok-1 + Hancock — AI Cybersecurity Agent

This repository combines the **Grok-1 314B Mixture-of-Experts** language model with
**Hancock**, CyberViser's AI-powered cybersecurity agent for pentesting, SOC analysis,
incident response, and more.

---

## 🚀 Quick Start

### 1. Install dependencies

```shell
pip install -r requirements.txt
```

### 2. Run the original Grok-1 demo

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` — see [Downloading the weights](#downloading-the-weights).

```shell
python run.py
```

### 3. Run the Hancock cybersecurity agent

**Interactive CLI (with Ollama backend):**

```shell
# Install and start Ollama first: https://ollama.com
export HANCOCK_LLM_BACKEND=ollama
python hancock_agent.py
```

**REST API server:**

```shell
export HANCOCK_API_KEY="replace-with-a-long-random-token"
python hancock_agent.py --server --port 5000
```

**With Grok-1 native backend (requires multi-GPU):**

```shell
python hancock_agent.py --backend grok
```

---

## 🤖 Hancock — Cybersecurity Modes

| Mode | Endpoint | Description |
|------|----------|-------------|
| 🔴 **Pentest** | `/v1/chat` (mode: pentest) | Recon, exploitation, CVE analysis, PTES reporting |
| 🔵 **SOC** | `/v1/triage`, `/v1/hunt`, `/v1/respond` | Alert triage, SIEM queries, PICERL incident response |
| ⚡ **Auto** | `/v1/chat` (mode: auto) | Context-aware pentest + SOC combined |
| 💻 **Code** | `/v1/code` | Security code: YARA, KQL, SPL, Sigma, Python |
| 👔 **CISO** | `/v1/ciso` | Compliance, risk reporting, board summaries |
| 🔍 **Sigma** | `/v1/sigma` | Sigma detection rule authoring |
| 🦠 **YARA** | `/v1/yara` | YARA malware detection rule authoring |
| 🔎 **IOC** | `/v1/ioc` | Threat intelligence enrichment |

### API Examples

**Alert Triage:**
```bash
curl -X POST http://localhost:5000/v1/triage \
  -H "Authorization: Bearer $HANCOCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"alert": "Mimikatz detected on DC01 at 03:14 UTC"}'
```

**Threat Hunting (Splunk):**
```bash
curl -X POST http://localhost:5000/v1/hunt \
  -H "Authorization: Bearer $HANCOCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"target": "lateral movement via PsExec", "siem": "splunk"}'
```

**Sigma Rule Generation:**
```bash
curl -X POST http://localhost:5000/v1/sigma \
  -H "Authorization: Bearer $HANCOCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"description": "Detect LSASS memory dump", "logsource": "windows sysmon", "technique": "T1003.001"}'
```

**YARA Rule Generation:**
```bash
curl -X POST http://localhost:5000/v1/yara \
  -H "Authorization: Bearer $HANCOCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"description": "Cobalt Strike beacon default HTTP profile", "file_type": "PE"}'
```

**IOC Enrichment:**
```bash
curl -X POST http://localhost:5000/v1/ioc \
  -H "Authorization: Bearer $HANCOCK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"indicator": "185.220.101.35", "type": "ip"}'
```

### CLI Commands

```
/mode pentest | soc | auto | code | ciso | sigma | yara | ioc
/clear          — clear conversation history
/history        — show history
/model <id>     — switch model
/exit           — quit
```

### Backends

| Backend | Env Var | Description |
|---------|---------|-------------|
| **grok** | `HANCOCK_LLM_BACKEND=grok` | Grok-1 native 314B MoE (requires GPU cluster) |
| **ollama** | `HANCOCK_LLM_BACKEND=ollama` | Local Ollama server (default) |
| **nvidia** | `HANCOCK_LLM_BACKEND=nvidia` + `NVIDIA_API_KEY` | NVIDIA NIM cloud API |
| **openai** | `HANCOCK_LLM_BACKEND=openai` + `OPENAI_API_KEY` | OpenAI API |

---

## 📋 File Structure

| File | Purpose |
|------|---------|
| `model.py` | Grok-1 model architecture (314B MoE, JAX) |
| `checkpoint.py` | Checkpoint loading and restoration |
| `runners.py` | Grok-1 inference runner and sampling |
| `run.py` | Original Grok-1 demo script |
| `hancock_agent.py` | Hancock cybersecurity agent (CLI + REST API) |
| `hancock_runner.py` | Bridge between Hancock and Grok-1 inference |
| `hancock_constants.py` | Shared constants and utilities |
| **`hancock_finetune.py`** | **Fine-tuning module for pentesting datasets** |
| **`hancock_dataset_collector.py`** | **Dataset collection and curation utilities** |
| **`hancock_adapter.py`** | **LoRA adapter management and integration** |
| `tests/test_hancock.py` | Unit tests for Hancock modules |

---

## 🎯 Fine-Tuning Grok-1 for Pentesting

**NEW:** This repository now includes comprehensive fine-tuning capabilities to specialize Grok-1 for pentesting and cybersecurity tasks using LoRA (Low-Rank Adaptation) for parameter-efficient training.

### Features

- ✅ **LoRA/QLoRA Adapter Support** — Efficient fine-tuning with minimal memory overhead
- ✅ **Pentesting Dataset Builder** — Generate training data from MITRE ATT&CK, CVE databases, and exploit techniques
- ✅ **SOC Training Data** — Include alert triage, SIEM queries, and incident response scenarios
- ✅ **Adapter Management** — Load and switch between specialized adapters at runtime
- ✅ **Multi-GPU Training** — Distributed training support for large-scale fine-tuning

### Quick Start: Fine-Tuning

#### 1. Prepare Training Data

Build a synthetic pentesting dataset from curated sources:

```bash
# Build complete pentesting + SOC dataset
python hancock_finetune.py --prepare-data --data-dir ./data/hancock-pentest

# Or collect from specific sources
python hancock_dataset_collector.py --collect mitre --output ./data/mitre
python hancock_dataset_collector.py --collect cve --output ./data/cve
python hancock_dataset_collector.py --collect all --output ./data/cybersec
```

This generates training examples covering:
- **MITRE ATT&CK** techniques, tactics, and procedures
- **CVE Analysis** for recent vulnerabilities (Log4Shell, XZ backdoor, etc.)
- **Exploitation Techniques** (SQL injection, XSS, command injection)
- **SOC Operations** (alert triage, Sigma rules, threat hunting)
- **Detection Engineering** (YARA rules, SIEM queries)

#### 2. Fine-Tune with LoRA

Train LoRA adapters on your pentesting dataset:

```bash
# Fine-tune Grok-1 with LoRA adapters
python hancock_finetune.py --train \
  --train-data ./data/hancock-pentest/train.jsonl \
  --val-data ./data/hancock-pentest/val.jsonl \
  --output ./adapters/hancock-pentest-v1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --epochs 3 \
  --learning-rate 2e-5

# With Weights & Biases logging
python hancock_finetune.py --train \
  --train-data ./data/hancock-pentest/train.jsonl \
  --val-data ./data/hancock-pentest/val.jsonl \
  --output ./adapters/hancock-pentest-v1 \
  --wandb --wandb-project my-hancock-finetune
```

**LoRA Parameters:**
- `--lora-r`: LoRA rank (default: 16, higher = more parameters but better performance)
- `--lora-alpha`: LoRA scaling factor (default: 32)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size per device (default: 1)

#### 3. Use Fine-Tuned Adapters

Load and use your fine-tuned adapters with Hancock:

```bash
# Use adapter with Grok-1 backend
python hancock_runner.py \
  --adapter ./adapters/hancock-pentest-v1 \
  --prompt "Explain CVE-2024-3094 exploitation techniques"

# Use adapter with Hancock agent
python hancock_agent.py --backend grok \
  --checkpoint ./checkpoints/ \
  --adapter ./adapters/hancock-pentest-v1
```

**Programmatic Usage:**

```python
from hancock_runner import HancockGrokRunner

# Initialize runner with fine-tuned adapter
runner = HancockGrokRunner(
    checkpoint_path="./checkpoints/",
    adapter_path="./adapters/hancock-pentest-v1"
)
runner.initialize()

# Generate responses with specialized knowledge
response = runner.generate(
    "What are the most effective techniques for post-exploitation on Windows?",
    max_len=512
)
print(response)
```

#### 4. Manage Multiple Adapters

Create and manage multiple specialized adapters:

```bash
# Create adapter config
python hancock_adapter.py --create-config \
  --name hancock-soc-v1 \
  --adapter-path ./adapters/hancock-soc-v1 \
  --specialization soc

# List all registered adapters
python hancock_adapter.py --list
```

**Adapter Specializations:**
- `pentest` — Penetration testing and offensive security
- `soc` — SOC analysis, alert triage, incident response
- `code` — Security code generation (YARA, Sigma, exploits)
- `ciso` — Risk management, compliance, executive reporting
- `general` — General cybersecurity knowledge

### Training Data Format

Training data should be in JSONL format with conversational structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Hancock, an elite penetration tester..."
    },
    {
      "role": "user",
      "content": "How do I test for SQL injection?"
    },
    {
      "role": "assistant",
      "content": "To test for SQL injection: 1) Try basic payloads like ' OR '1'='1'..."
    }
  ],
  "metadata": {
    "task_type": "exploitation",
    "category": "web_security"
  }
}
```

### Evaluation

Evaluate your fine-tuned models:

```bash
# Evaluate on test set
python hancock_finetune.py --evaluate \
  --checkpoint ./adapters/hancock-pentest-v1 \
  --test-data ./data/hancock-pentest/val.jsonl
```

### Dataset Sources

The fine-tuning pipeline can integrate data from:

1. **MITRE ATT&CK Framework** — Techniques, tactics, and detection methods
2. **CVE/NVD Databases** — Vulnerability analysis and exploitation
3. **Exploit Databases** — Real-world exploitation techniques
4. **Sigma Rules** — Detection engineering and SIEM queries
5. **CTF Write-ups** — Practical penetration testing scenarios
6. **Security Blogs** — Latest threat intelligence and techniques

### Best Practices

- **Start with synthetic data** using `hancock_finetune.py --prepare-data`
- **Validate quality** by reviewing generated examples before training
- **Use smaller LoRA rank** (r=8-16) to start, increase if underfitting
- **Monitor training** with Weights & Biases or TensorBoard
- **Test adapters** on validation set before deployment
- **Version your adapters** with clear naming (e.g., `hancock-pentest-v2-mitre-cve`)

### Hardware Requirements

**Fine-Tuning:**
- Minimum: 1x A100 80GB or 2x A6000 48GB
- Recommended: 4x A100 80GB for faster training
- LoRA reduces memory requirements significantly vs full fine-tuning

**Inference with Adapters:**
- Same as base Grok-1 model (8x GPUs minimum)
- Adapters add <1% overhead to inference time

---

# Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):
```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# Running Tests

```shell
pip install flask openai
python -m unittest tests.test_hancock -v
```

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.
