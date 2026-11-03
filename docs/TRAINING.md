# Training pipeline

All commands run from the repository root. Paths follow `configs/`.

## 1. Data

```bash
amp-prioritizer data verify                 # check data/ against manifest.json
python scripts/fetch_negatives.py           # AMPlify non-AMP set (not bundled)
```

Genome k-mer features are shipped in `data/current/`. To rebuild them from RefSeq:

```bash
python scripts/download_genomes.py \
    --species_lists data/current/top_species.json --out data/current/genomes
python -m amp_prioritizer.species_genome build-features \
    --species_lists data/current/top_species.json \
    --genome_dir data/current/genomes \
    --out data/current/species_kmer_features.npz
```

## 2. Predictor

```bash
accelerate launch --num_processes 8 --mixed_precision fp16 \
    -m amp_prioritizer.train_stripped --config configs/benchmark.yaml

accelerate launch --num_processes 8 --mixed_precision fp16 \
    -m amp_prioritizer.train --config configs/full.yaml --stage 2b
```

For a quick single-GPU check add `--max_steps 200`.

## 3. Optional MLM adaptation

```bash
accelerate launch --num_processes 8 --mixed_precision fp16 \
    -m amp_prioritizer.mlm_pretrain --config configs/stage0_mlm.yaml
python -m amp_prioritizer.merge_stage0_lora \
    --ckpt runs/stage0_mlm/best.pt --out runs/stage0_merged
```

Then set `model.name: runs/stage0_merged` in the predictor configuration. Without this step
the predictors start from the public ESM-2 weights.

## 4. Generator

```bash
python scripts/cache_latents.py --config configs/full.yaml --ckpt runs/full/best.pt
python -m amp_prioritizer.gen.train_gen --config configs/stage3_gen.yaml \
    --cache data/current/latent_cache.npz
```

`configs/sft_template.yaml` holds the target-specific fine-tuning settings and the
candidate sampling and screening parameters.

## 5. Scoring

See the *Scoring candidate peptides* section of the README.
