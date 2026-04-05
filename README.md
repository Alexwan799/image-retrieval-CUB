# Image Retrieval on CUB-200-2011

Fine-grained image retrieval experiments on CUB-200-2011 using a ResNet50 embedding model, ProxyNCA loss, PK sampling, and retrieval metrics such as Recall@K and mAP.

## Current Best Result

- Dataset: CUB-200-2011
- Backbone: ResNet50 (ImageNet pretrained)
- Embedding dim: 256
- Loss: ProxyNCA
- Scheduler: StepLR (`step_size=10`, `gamma=0.2`)
- Sampler: `p=16`, `k=2`
- mAP: `0.6442`
- Recall@1: `0.7206`

## Repository Layout

- `src/`: training, evaluation, model, dataset, sampler, and visualization code
- `configs/default.yaml`: default training recipe
- `scripts/run_sweep.py`: local sweep entry point
- `notebooks/`: experiment analysis and failure-case analysis notebooks
- `doc/`: notes

Large experiment artifacts are intentionally excluded from git:

- `data/`
- `checkpoints/`
- `results/`

## Setup

Recommended: Python 3.10+.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

This project expects the CUB-200-2011 dataset under:

```bash
data/CUB_200_2011
```

The default config points to:

```yaml
data:
  root: data/CUB_200_2011
```

## Train

Run training with the default config:

```bash
python src/train.py
```

Run with a specific config:

```bash
python src/train.py --config configs/default.yaml
```

Resume from a checkpoint:

```bash
python src/train.py --resume checkpoints/<run_id>/best.pth
```

## Evaluate

Evaluate a checkpoint:

```bash
python src/evaluate.py --checkpoint checkpoints/<run_id>/best.pth
```

Save failure cases during evaluation:

```bash
python src/evaluate.py --checkpoint checkpoints/<run_id>/best.pth --save_fail
```

## Query Visualization

Visualize nearest retrieved images for a query:

```bash
python src/visualise.py --checkpoint checkpoints/<run_id>/best.pth --q_image path/to/query.jpg --q_num 5
```

Query figures are saved under:

```bash
results/query_results/
```

## Sweeps

Run the local sweep script:

```bash
python scripts/run_sweep.py
```

Sweep summaries are written to:

```bash
results/sweeps/
```

## Notes

- The current default config already reflects the strongest recipe found so far.
- Training-time validation uses retrieval mAP rather than loss-only validation.
- The notebooks are templates and expect local experiment artifacts to exist under `results/` and `checkpoints/`.
