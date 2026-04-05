# Image Retrieval on CUB-200-2011

This is my bird image retrieval project on CUB-200-2011. I started with a simple Triplet Loss baseline, then tried ProxyNCA, better preprocessing, and different PK sampler settings, and finally got the current best result below.

## Current Best Result

- Dataset: CUB-200-2011
- Backbone: ResNet50 (ImageNet pretrained)
- Embedding dim: 256
- Loss: ProxyNCA
- Scheduler: StepLR (`step_size=10`, `gamma=0.2`)
- Sampler: `p=16`, `k=2`
- mAP: `0.6442`
- Recall@1: `0.7206`

## Experiments

This table does not list every single run. It only keeps the main turning points, so it is easier to see how the project moved from the first baseline to the current best setup.

| Setting | Loss | embed_dim | lr | mAP | Recall@1 |
| --- | --- | --- | --- | --- | --- |
| Triplet baseline | Triplet | 128 | 1e-4 | 0.5474 | 0.7095 |
| First strong ProxyNCA run | ProxyNCA | 64 | 1e-4 | 0.6022 | 0.6938 |
| Better overall ranking | ProxyNCA | 128 | 1e-4 | 0.6148 | 0.7047 |
| Best pre-normalization recipe | ProxyNCA | 256 | 1e-4 | 0.6200 | 0.6949 |
| Add ImageNet normalization | ProxyNCA + scheduler | 256 | 1e-4 | 0.6284 | 0.7061 |
| Current best (`p=16, k=2`) | ProxyNCA + scheduler | 256 | 1e-4 | **0.6442** | **0.7206** |

What mattered most:

- Switching from Triplet to ProxyNCA gave the biggest improvement.
- Adding proper ImageNet normalization also helped a lot.
- PK sampler settings mattered more than I expected. `p=16, k=2` worked much better than the older `8x4` setup.

## Failure Case Analysis

The model still makes mistakes in a few common ways:

1. Background or scene bias: sometimes the model pays too much attention to the environment. A Rufous Hummingbird near a red feeder can get matched to red tanager-like birds instead.
2. Pose and crop changes: even for the same species, very different body pose or image framing can make retrieval worse.
3. Very similar species: some bird classes are just extremely close visually, so confusion is still common.

Some example retrieval results from the current best checkpoint:

| Case | Example |
| --- | --- |
| Mostly correct retrieval | ![White-breasted nuthatch success](assets/readme/success_white_breasted_nuthatch.png) |
| Good retrieval with one near miss | ![Olive-sided flycatcher success](assets/readme/success_olive_sided_flycatcher.png) |
| Background/color bias example | ![Rufous hummingbird to tanager confusion](assets/readme/failure_context_color_bias.png) |
| Similar seabird confusion | ![Albatross confusion](assets/readme/failure_albatross_confusion.png) |
| Bad failure case | ![Cuckoo to waxwing confusion](assets/readme/failure_cuckoo_waxwing_confusion.png) |

## Repository Layout

- `src/`: training, evaluation, model, dataset, sampler, and visualization code
- `configs/default.yaml`: default training recipe
- `scripts/run_sweep.py`: local sweep entry point
- `notebooks/`: failure-case analysis notebook

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
python src/train.py --config <config_path>
```

Resume from a checkpoint:

```bash
python src/train.py --resume <checkpoint_path>
```

## Evaluate

Evaluate a checkpoint:

```bash
python src/evaluate.py --checkpoint <checkpoint_path>
```

Save failure cases during evaluation:

```bash
python src/evaluate.py --checkpoint <checkpoint_path> --save_fail
```

## Query Visualization

Visualize nearest retrieved images for a query:

```bash
python src/visualise.py --checkpoint <checkpoint_path> --q_image <query_image_path> --q_num 5
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
- Typical values:
  - `<config_path>`: `configs/default.yaml`
  - `<checkpoint_path>`: `checkpoints/<run_id>/best.pth`
