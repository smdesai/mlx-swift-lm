# Batch Generation Test Models (≤7B parameters)

Models mapped from `LLMModelFactory` registry to smallest available 4-bit (or bf16) instruct models on HuggingFace.

| # | Model Type | Model ID | Size | Notes |
|---|-----------|----------|------|-------|
| 1 | `llama` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 1B | Also covers `mistral` type |
| 2 | `phi3` | `mlx-community/Phi-3.5-mini-instruct-4bit` | 3.8B | |
| 3 | `gemma` | `mlx-community/quantized-gemma-2b-it` | 2B | |
| 4 | `gemma2` | `mlx-community/gemma-2-2b-it-4bit` | 2B | |
| 5 | `gemma3_text` | `mlx-community/gemma-3-1b-it-qat-4bit` | 1B | Also covers `gemma3` type |
| 6 | `gemma3n` | `mlx-community/gemma-3n-E2B-it-lm-4bit` | 2B eff | |
| 7 | `qwen2` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 1.5B | Also covers `acereason` type |
| 8 | `qwen3` | `mlx-community/Qwen3-0.6B-4bit` | 0.6B | Uses --no-think |
| 9 | `granite` | `mlx-community/granite-3.3-2b-instruct-4bit` | 2B | |
| 10 | `bitnet` | `mlx-community/bitnet-b1.58-2B-4T-4bit` | 2B | |
| 11 | `smollm3` | `mlx-community/SmolLM3-3B-4bit` | 3B | |
| 12 | `ernie4_5` | `mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft` | 0.3B | Pretrain model, bf16 |
| 13 | `lfm2` | `mlx-community/LFM2-1.2B-4bit` | 1.2B | |
| 14 | `exaone4` | `mlx-community/exaone-4.0-1.2b-4bit` | 1.2B | |
| 15 | `lille-130m` | `mlx-community/lille-130m-instruct-bf16` | 0.13B | bf16 |
| 16 | `openelm` | `mlx-community/OpenELM-270M-Instruct` | 0.27B | |
| 17 | `olmo2` | `mlx-community/OLMo-2-1124-7B-Instruct-4bit` | 7B | |
| 18 | `olmoe` | `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` | 7B (1B active) | MoE |
| 19 | `mimo` | `mlx-community/MiMo-7B-SFT-4bit` | 7B | |
| 20 | `cohere` | `mlx-community/c4ai-command-r7b-12-2024-4bit` | 7B | |
| 21 | `internlm2` | `mlx-community/internlm2_5-7b-chat-4bit` | 7B | |
| 22 | `falcon_h1` | `mlx-community/Falcon-H1-3B-Instruct-4bit` | 3B | SSM hybrid |
| 23 | `jamba_3b` | `mlx-community/AI21-Jamba-Reasoning-3B-bf16` | 3B | SSM hybrid, bf16 |
| 24 | `nanochat` | `dnakov/nanochat-d20-mlx` | tiny | Not mlx-community |

## Architectures skipped (>7B or unavailable)

| Model Type | Reason |
|-----------|--------|
| `deepseek_v3` | 671B - far too large |
| `phimoe` | 42B total |
| `mistral3` | 24B (Mistral-Small-3.x) |
| `qwen3_next` | 80B total |
| `qwen3_5` | 27B |
| `qwen3_5_moe` | 35B total |
| `qwen3_moe` | 30B total |
| `olmo3` | 32B |
| `gpt_oss` | 20B+ |
| `minimax` | 46B MoE |
| `mimo_v2_flash` | 309B total |
| `bailing_moe` | 16B MoE |
| `granitemoehybrid` | ~7B hybrid (only 3-bit available) |
| `apertus` | 8B |
| `baichuan_m1` | 14B |
| `glm4` | 9B |
| `glm4_moe` | Large MoE |
| `glm4_moe_lite` | Large MoE |
| `nemotron_h` | No MLX conversion available |
| `afmoe` | No MLX conversion available |
| `lfm2_moe` | 8B total (borderline) |
| `minicpm` | No mlx-community 4-bit |
| `phi` | Base model only (phi-2), no instruct |
| `starcoder2` | Code completion, not instruct |
