# EMFormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting (ICML 2026)

![HKUST](assets/HKUST-logo.png)

<p align="center">
  <a href='https://openreview.net/pdf?id=cKztWFFGNE'><img src='https://img.shields.io/static/v1?label=Paper&message=EMFormer&color=red&logo=openreview'></a>
  &nbsp;
  <a href='https://github.com/chenhao-zju/emformer'><img src='https://img.shields.io/badge/Code-GitHub-blue?logo=github'></a>
</p>

EMFormer is a global weather forecasting model (EMTransformer) trained and
evaluated through a single YAML-driven pipeline (`train.py` / `test.py`).

## 1. What This Repository Supports

- End-to-end training and evaluation for global forecasting
- YAML-driven experiment configuration
- DDP multi-GPU training with AMP mixed precision
- Multi-step rollout training (`t_out_train`) and iterative inference
- Checkpointing and log output under `./logs/`

## 2. Repository Layout

```text
emformer/
|- config/
|  `- EMFormer.yaml
|- networks/
|  |- EMFormer.py
|  |- l2_loss.py
|  `- __init__.py
|- utils/
|  |- data_loader_npyfiles.py
|  |- weighted_acc_rmse.py
|  |- metrics.py
|  |- YParams.py
|  |- logging_utils.py
|  `- fileio.py
|- train.py
|- test.py
|- inference.py
|- train.sh
|- test.sh
`- logs/
```

## 3. Environment Setup

### 3.1 Python and CUDA

- Python: 3.8+
- OS: Linux recommended for multi-GPU training
- GPU: CUDA-capable GPU(s)

### 3.2 Core Dependencies

Install the commonly used dependencies first:

```bash
pip install numpy scipy einops ruamel.yaml tqdm timm wandb
```

Install PyTorch matching your CUDA version (example shown for CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional:

- `apex` (for `FusedAdam`); the code falls back gracefully if not installed.
- `wandb` (for experiment logging); disabled unless `log_to_wandb: true`.

## 4. Data Preparation

### 4.1 Required Statistics Files

Normalization statistics are read from `{root_dir}/statistic/`:

- `{root_dir}/statistic/mean_std.json` (pressure-level variables)
- `{root_dir}/statistic/mean_std_single.json` (surface variables)

### 4.2 Dataset Organization (From `utils/data_loader_npyfiles.py`)

Data is read by timestamp. Each variable is stored as a separate `.npy` file.

Time path format:

```text
{YYYY}/{YYYY-MM-DD}/
```

Filename format:

```text
{HH}:00:00-{variable}.npy                 # surface variable
{HH}:00:00-{variable}-{pressure_level}.npy  # pressure-level variable
```

Expected layout:

- Pressure-level variables:
  - `{root_dir}/{YYYY}/{YYYY-MM-DD}/{HH}:00:00-{var}-{level}.npy`
- Surface variables:
  - `{root_dir}/single/{YYYY}/{YYYY-MM-DD}/{HH}:00:00-{var}.npy`

Default variables used by the loader:

- Pressure-level (`higher_features`): `z`, `q`, `u`, `v`, `t`
- Surface (`surface_features`): `t2m`, `u10`, `v10`, `msl`, `sp`

Default pressure levels (13 levels):

```text
1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0
```

This gives `5 vars x 13 levels + 5 surface = 70` channels, matching the default
`feature_dims: 70`.

Directory example:

```text
<root_dir>/
|- statistic/
|  |- mean_std.json
|  `- mean_std_single.json
|- 2021/
|  `- 2021-01-09/
|     |- 00:00:00-z-1000.0.npy
|     |- 00:00:00-z-925.0.npy
|     |- 00:00:00-q-1000.0.npy
|     |- ...
|     `- 00:00:00-t-50.0.npy
`- single/
   `- 2021/
      `- 2021-01-09/
         |- 00:00:00-t2m.npy
         |- 00:00:00-u10.npy
         |- 00:00:00-v10.npy
         |- 00:00:00-msl.npy
         `- 00:00:00-sp.npy
```

### 4.3 Climate Files (Evaluation Only)

For `valid` / `test` runs the loader also reads daily climatology under
`root_dir`:

- `{root_dir}/climate_mean_day_128x256/1993-2016/` (pressure-level climate)
- `{root_dir}/single/climate_mean_day_128x256/1993-2016/` (surface climate)

These are only needed when computing ACC during evaluation; training does not
use them.

## 5. Configuration

The model is configured through `config/EMFormer.yaml`. The `--config` argument
selects the section name inside the file (`EMFormer`).

### 5.1 Key Fields You Will Usually Modify

- `root_dir` — dataset root
- `train_period`, `valid_period`, `test_period` — `[start_year, end_year]`
- `batch_size`
- `feature_dims`, `in_channels`, `out_channels`
- `h_size`, `w_size`, `ori_h_size`, `ori_w_size`, `patch_size`
- `embed_dim`, `encoder_depths`
- `t_out_train` — rollout steps used during training
- `max_epochs`, `lr`, `min_lr`, `scheduler`

## 6. Training

### 6.1 Script Entry

`train.sh` runs the EMFormer config with 4-GPU DDP by default:

```bash
bash train.sh
```

### 6.2 Manual Equivalent

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
  --enable_amp \
  --yaml_config=./config/EMFormer.yaml \
  --config=EMFormer \
  --run_num=1 \
  --exp_dir=./logs/your_exp/ \
  --checkpoint=''
```

## 7. Testing

### 7.1 Script Entry

```bash
bash test.sh
```

### 7.2 Manual Equivalent

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --yaml_config=./config/EMFormer.yaml \
  --config=EMFormer \
  --run_num=1 \
  --override_dir=./logs/your_eval/ \
  --weights=./logs/your_exp/EMFormer/1/training_checkpoints/best_ckpt.tar
```

`--override_dir` and `--weights` must be used together: outputs are written to
`--override_dir` and weights are loaded from `--weights`.

## 8. Outputs and Logs

By default, each experiment writes to:

- `./logs/<exp_name>/<config>/<run_num>/`

Common artifacts:

- `out.log` (training log)
- `training_checkpoints/ckpt.tar`
- `training_checkpoints/best_ckpt.tar`
- `hyperparams.yaml`

Evaluation saves metric arrays (RMSE/ACC) and, when enabled, raw forecasts
(`pred.npy`, `label.npy`) into the experiment / override directory, controlled
by `save_rmse_acc` and `save_raw_forecasts` in the config.

## 9. Reproducibility Tips

- Keep `config/*.yaml` and shell scripts versioned together.
- Fix random seeds in training code if strict reproducibility is required.
- Record CUDA, PyTorch, and dependency versions in each run log.
- Use explicit checkpoint paths when comparing experiments.

## 10. Troubleshooting

- Missing statistics files:
  - Ensure `mean_std.json` and `mean_std_single.json` exist under
    `{root_dir}/statistic/`.
- DDP launch issues:
  - Verify `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and NCCL compatibility.

## 11. License

This repository is distributed under the BSD 3-Clause License.
See `LICENSE` for details.

## 12. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
chen2026emformer,
title={{EMF}ormer: Efficient Multi-Scale Transformer for Accumulative Context Weather Forecasting},
author={Hao Chen and Tao Han and Jie ZHANG and Song Guo and Fenghua Ling and LEI BAI},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=cKztWFFGNE}
}
```
