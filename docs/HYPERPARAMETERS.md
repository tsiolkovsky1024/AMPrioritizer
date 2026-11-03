# Hyperparameters

Reference values for the configurations in [`../configs/`](../configs/). The effective
batch size is `per_device_batch_size × grad_accum × n_gpus` (8 GPUs in the reference runs:
8 × 2 × 8 = 128).

## Predictor (`configs/benchmark.yaml`, `configs/full.yaml`)

| Setting | Value |
|---|---|
| Backbone | `facebook/esm2_t33_650M_UR50D`, full fine-tuning, mean pooling |
| Precision | FP16, gradient checkpointing |
| Epochs | 5 |
| Optimiser | AdamW, weight decay 0.01, gradient clip 1.0 |
| Learning rate | backbone 1e-5, heads 1e-4, warm-up ratio 0.05 |
| Head width / dropout | 512 / 0.3 |
| Organism encoder | genome 4-mer features (340-d) → MLP (hidden 512) |
| Max sequence length | 200 |
| Active threshold | log10 MIC (µM) ≤ 1.5 |
| Label smoothing | 0.2 |
| Seeds | benchmark: 2 and 3; full: 2 |

### Loss weights

| Term | Benchmark | Full (phase 1) | Full (phase 2) |
|---|---|---|---|
| AMP classification | – | 1.0 | 0.2 |
| Species activity | 0.5 | 0.5 | 0.25 |
| Censored (Tobit) MIC likelihood | 0.3 | 0.3 | 0.2 |
| Huber (activity) | 0.5 | 0.5 | 0.5 |
| LambdaRank | 0.5 | 0.3 | 0.3 |
| ListMLE | 0.2 | 0.2 | 0.1 |
| Physicochemical | – | 0.1 | 0.1 |
| Hemolysis | – | 0.2 | 0.2 |
| Selectivity | – | 0.01 | 0.01 |
| Consistency (symmetric KL) | – | 0.01 | 0.01 |

The full family switches from phase 1 to phase 2 after 40 % of the training steps
(`train.curriculum.switch_fraction`). The benchmark family uses the `gap_pairwise` ranking
loss with NDCG cut-off 50. Negatives are drawn from the AMPlify non-AMP set at a 1:1 ratio
and take no part in the MIC losses.

## Candidate generator

| Setting | Stage 3 pre-training (`configs/stage3_gen.yaml`) | Target SFT (`configs/sft_template.yaml`) |
|---|---|---|
| Architecture | latent CVAE (z = 128), 4-layer AR Transformer, width 512, length head | same, initialised from Stage 3 |
| Conditioning input | 1280-d pooled predictor embedding | – |
| Epochs | 30 | 4 |
| Batch size | 64 | 32 |
| Learning rate | 3e-4 | 2e-4 |
| Weight decay / warm-up | 0.01 / 0.05 | 0.01 / 0.05 |
| KL schedule | cyclic β (period 10,000 steps, max 0.5), free bits 0.5 nat/dim | – |
| Candidate pool / kept | – | 5,000 sampled, top 500 by predicted MIC |
| Anchor MIC window | – | 0.1–32 µM, anchor weight 2.0 |
| Sampling | – | nucleus p = 0.95, temperature 1.0, length 5–60 |

## Optional Stage 0 adaptation (`configs/stage0_mlm.yaml`)

Masked-language-model adaptation of ESM-2 650M with LoRA (r = 16, α = 32, dropout 0.05) for
2 epochs, learning rate 1e-4, mask ratio 0.15 with span 3. `amp_prioritizer.merge_stage0_lora`
merges the adapters into a plain Hugging Face directory that can be passed as `model.name`.
