# Paper 2-bit VPTQ presets

Source: [arXiv:2409.17066v2, Appendix B, Table 8](https://arxiv.org/html/2409.17066v2#A2).
The default target is the local **Llama-3.1-8B-Instruct** checkpoint. Its
`llama31-8b-instruct-2bit` preset transfers the original LLaMA-3 8B 2.08-bit
row's parameters: `N%=1, v0=4, k0=4096, v1=12, k1=k2=4096, groups=1`.
The 2.08-bit value belongs to the source model in the paper; the target model's
effective bitwidth must be measured after quantization.

Default paths:

```text
Model:   /data/LLMs/Llama-3.1-8B-Instruct
Hessian: /data/Hessians/Llama-3.1-8B-Instruct-6144-8k
Inverse: /data/Hessians/Llama-3.1-8B-Instruct-6144-8k/InvHessians-Llama-31-8B-Instruct-6144-8k
```

The supplied Hessian path contained the same absolute directory twice;
the script uses a single copy. `SEQ_LEN` defaults to 8192 for this preset.
The original paper presets remain available below.

| Preset | Paper effective bits | Outlier N% | v0 | k0 | v1 | k1 | k2 | Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama2-7b-2.02 | 2.02 | 0 | -1 | -1 | 6 | 4096 | -1 | 1 |
| llama2-7b-2.26 | 2.26 | 1 | 4 | 8192 | 12 | 4096 | 4096 | 4 |
| llama2-13b-2.02 | 2.02 | 0 | -1 | -1 | 6 | 4096 | -1 | 1 |
| llama2-13b-2.18 | 2.18 | 2 | 4 | 8192 | 12 | 4096 | 4096 | 4 |
| llama2-70b-2.07 | 2.07 | 1 | 4 | 8192 | 12 | 4096 | 4096 | 4 |
| llama2-70b-2.11 | 2.11 | 1 | 4 | 8192 | 12 | 4096 | 4096 | 8 |
| llama3-8b-2.08 | 2.08 | 1 | 4 | 4096 | 12 | 4096 | 4096 | 1 |
| llama3-8b-2.24 | 2.24 | 1 | 4 | 8192 | 6 | 4096 | -1 | 16 |
| llama3-70b-2.02 | 2.02 | 0 | -1 | -1 | 12 | 4096 | 4096 | 1 |
| llama3-70b-2.07 | 2.07 | 1 | 4 | 4096 | 6 | 4096 | -1 | 16 |

## Run

Activate the Linux/WSL `vptq-algo` environment from `algo-environment.yml`,
with the additional dependencies described in `algorithm.md`, and install this
checkout of VPTQ. Use `SKIP_COMPILE=1 pip install -e . --no-build-isolation` for
the Torch dequantization fallback, or compile the CUDA extension for fast inference.
The default model must already be present at the local path above.
For a Hugging Face model preset, obtain access before launching.

```bash
cd VPTQ

# Print the local Llama-3.1 command; no GPU, Python or data files needed.
bash scripts/quantize_2bit.sh --dry-run

# Quantize local Llama-3.1-8B-Instruct using the configured paths.
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
bash scripts/quantize_2bit.sh

# Same configuration, with four GPUs and disk-based layer transfer.
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
bash scripts/quantize_2bit.sh
```

Run from any directory; the script changes to its own repository root.
Relative Hessian/output paths are therefore relative to `VPTQ/`.
Absolute paths are recommended for data. Output is written to
`outputs/paper-2bit/<preset>/<timestamp>/packed_model/` by default.
For the default target this is
`outputs/paper-2bit/llama31-8b-instruct-2bit/<timestamp>/packed_model/`.
The runner also evaluates WikiText-2 and C4-new and writes `ppl_results.json`.
It currently evaluates at sequence length 2048 regardless of `SEQ_LEN`.
The JSON includes a `quantization_config` object below the perplexity results,
recording the model name, vector/codebook sizes, outlier percentage, groups,
K-means and normalization settings, seed, quantization sequence length,
Hessian paths and packed-model setting. Vector/codebook lists use
`[outlier, main]` order, with `-1` for disabled entries.

## Parameter mapping

- `--vector_lens v0 v1`: outlier/main vector lengths along output channels.
- `--num_centroids k0 k1`: outlier/main codebook sizes.
- `--num_res_centroids -1 k2`: no outlier residual codebook; optional main residual codebook.
- `--npercent N`: percentage of input columns assigned to the outlier codebook.
- `--group_num`: number of main codebook groups; `--group_size -1` lets the code derive the size.
- `-1` means disabled for the corresponding outlier/residual setting.

For ordinary columns, indices cost `log2(k1)/v1` bits per weight, plus
`log2(k2)/v1` if residual quantization is enabled. Thus `v1=6,k1=4096`
and `v1=12,k1=k2=4096` both use 2 index bits per weight.
Codebooks and higher-precision outlier indices add to the effective bitwidth.

The remaining settings follow this checkout's tutorial/defaults, not Table 8:
Hessian-weighted K-means, `kiter=100`, `ktol=1e-5`, permutation enabled,
min/max weight normalization enabled with `norm_dim=0`, block size 128,
quantization step 1, damping 0.01, seed 0, and packed checkpoint saving.
The current layer constructor does not forward CLI block size/step/damping;
the script passes values matching the constructor defaults.
Multi-GPU runs additionally use `--save_qlinear` for disk-based layer transfer.
Single-GPU runs omit it because the current runner does not reload those files.

## Reproduction limits

- Table 8 reports effective bits per Transformer block, including its seven
  Linear operators. It does not specify the size of the whole serialized model.
  Embeddings/lm_head, normalization metadata, padding and serialization overhead
  can change the actual checkpoint bits per original parameter.
- Hessian and precomputed inverse/factor files must match the exact model and
  the preprocessing/permutation expected by this checkout. The script checks
  file presence, not their numeric compatibility. Use files containing
  `flatH/n/mu` and `invH/perm/zero_idx`, respectively. This repository does not
  provide their collection script; both paths are required by the current flow.
  `SEQ_LEN` does not change the calibration data already used to create them.
- The paper uses 128 C4 segments for calibration. Matching that experiment
  requires statistics collected from the same calibration setup.
- LLaMA-3.1, LLaMA-3.2 and Qwen are not original Table 8 model presets.
  The default Llama-3.1 preset transfers a configuration from LLaMA-3 8B;
  it does not establish the paper's reported bitwidth or accuracy for this
  checkpoint. The same applies when overriding `MODEL_NAME`.
- Although Table 8's title mentions Mistral, the published table lists only
  LLaMA-2/3 rows. No Mistral settings are invented here.
- Layer-wise/end-to-end fine-tuning from the paper is not performed by this
  released runner. These scripts reproduce the quantization settings, not the
  complete fine-tuned results. Outlier and multi-group CUDA paths are labeled
  untested in `algorithm.md`; verify a resulting checkpoint before benchmarking.
- A one-line correction in `NPVectorQuantizer.init_res_centroids_indices()`
  reduces residual K-means weights to one scalar per vector, matching the main
  K-means path and the paper's weighted initialization.
