# Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models

[![Project](https://img.shields.io/badge/Project-Page-green)](https://pabloruizponce.com/papers/Interact2Ar)
[![arXiv](https://img.shields.io/badge/arXiv-2512.19692-b31b1b.svg)](https://arxiv.org/abs/2512.19692)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-5c4ee5.svg)](https://docs.astral.sh/uv/)

![Interact2Ar cover](https://files.pabloruizponce.com/papers/Interact2Ar/cover.png)

## About

Generating realistic human-human interactions is a challenging task that requires not only high-quality individual body and hand motions, but also coherent coordination among all interactants. Due to limitations in available data and increased learning complexity, previous methods tend to ignore hand motions, limiting the realism and expressivity of the interactions. Additionally, current diffusion-based approaches generate entire motion sequences simultaneously, limiting their ability to capture the reactive and adaptive nature of human interactions. To address these limitations, we introduce Interact2Ar, the first end-to-end text-conditioned autoregressive diffusion model for generating full-body, human-human interactions. Interact2Ar incorporates detailed hand kinematics through dedicated parallel branches, enabling high-fidelity full-body generation. Furthermore, we introduce an autoregressive pipeline coupled with a novel memory technique that facilitates adaptation to the inherent variability of human interactions using efficient large context windows. The adaptability of our model enables a series of downstream applications, including temporal motion composition, real-time adaptation to disturbances, and extension beyond dyadic to multi-person scenarios. To validate the generated motions, we introduce a set of robust evaluators and extended metrics designed specifically for assessing full-body interactions. Through quantitative and qualitative experiments, we demonstrate the state-of-the-art performance of Interact2Ar.

## News

- [20/02/2026] Interact2Ar is accepted at CVPR 2026.
- [22/12/2025] The paper is available on [arXiv](https://arxiv.org/abs/2512.19692).

## TODO List

- [x] Release main-paper training, inference, evaluation, and visualization code.
- [x] Release model and evaluator checkpoint download link.
- [x] Add license.

## Installation

Install [uv](https://docs.astral.sh/uv/) and clone the repository:

```bash
git clone https://github.com/pabloruizponce/Interact2Ar.git
cd Interact2Ar
```

Install the locked dependencies:

```bash
uv sync
```

Install the visualization extra if you want to render videos with `aitviewer`:

```bash
uv sync --extra visualization
```

## Data

Interact2Ar uses the Inter-X dataset and SMPL-X body model files. Download them from their original sources before running training or evaluation:

- [Inter-X project page](https://liangxuy.github.io/inter-x/) and [official Inter-X repository](https://github.com/liangxuy/Inter-X) for the dataset, processed data, split files, and Inter-X data license.
- [SMPL-X official download page](https://smpl-x.is.tue.mpg.de/download.php) for `SMPLX_NEUTRAL.npz` and the SMPL-X license.

> [!IMPORTANT]
> Inter-X and SMPL-X are not redistributed in this repository. Please follow the original download forms, licenses, and citation requirements from each project.

After downloading the data, create the repository layout under `data/`:

```bash
uv run python scripts/prepare_data_layout.py \
  --source-root /path/to/InterX \
  --smplx-neutral /path/to/SMPLX_NEUTRAL.npz
```

By default this creates symlinks. Add `--mode copy` for a standalone copy, and add `--force` only when replacing existing files intentionally.

Expected layout:

```text
data/
  splits/
    train.txt
    val.txt
    test.txt
    all.txt
  motions/
    train.h5
    val.h5
    test.h5
  joints/                  # optional cache; losses can recompute joints from SMPL-X
    train.h5
    val.h5
    test.h5
  texts_processed/
    <motion_id>.txt
  texts_tensors/           # optional cache; CLIP can run on the fly
    <motion_id>.npy
  glove/
    hhi_vab_data.npy
    hhi_vab_idx.pkl
    hhi_vab_words.pkl
  body_models/
    smplx/
      SMPLX_NEUTRAL.npz
```

## Checkpoints

Download the released Interact2Ar model checkpoint and evaluator weights from [Google Drive](https://drive.google.com/drive/folders/1SxfyL29SIwS80GBByYoXrmO_9eMveuNE?usp=sharing).

Place the downloaded files in this layout:

```text
release-assets/
  checkpoints/
    interact2ar_mixed_memory.ckpt
  checkpoints-evaluator/
    joints_full/model/finest.tar
    joints_body/model/finest.tar
    joints_hands/model/finest.tar
```

Then link or copy the assets into the repository:

```bash
uv run python scripts/prepare_release_assets.py \
  --source-root release-assets \
  --check-hash
```

Use `--mode copy` instead of the default symlink mode when preparing a self-contained folder. The manifest in `assets/checkpoints.json` records the expected names, SHA-256 hashes, sizes, and key hyperparameters.

## Training

Train the main-paper autoregressive Interact2Ar model with the Python entry point:

```bash
uv run python src/train.py \
  --dataset paper.yaml \
  --diffusion ar/paper.yaml \
  --denoiser paper.yaml \
  --train paper.yaml \
  --run-name interact2ar_paper \
  --seed 44 \
  --gpu "[0]"
```

For multiple GPUs, pass a list such as `--gpu "[0,1,2,3]"`. Training writes checkpoints to `checkpoints/<run-name>/` and TensorBoard logs to `runs/<run-name>/`.

## Inference

Generate samples with the released checkpoint:

```bash
uv run python src/predict.py \
  --checkpoint checkpoints/interact2ar_mixed_memory.ckpt \
  --text "One person waves at the other person." \
  --length 150 \
  --deterministic
```

Generated tensors are written to `outputs/inference/`. Add `--novis` to skip video rendering. Rendering requires `uv sync --extra visualization` and a working `aitviewer` setup.

> [!NOTE]
> In headless Ubuntu Docker containers, install `libegl1 libegl-dev libgles2 libgl1 ffmpeg` and run rendering commands with `INTERACT2AR_RENDER_BACKEND=egl`.

## Evaluation

Run the main-paper text-motion metrics with the three joint evaluators:

```bash
uv run python src/evaluate.py \
  --dataset paper.yaml \
  --checkpoint checkpoints/interact2ar_mixed_memory.ckpt \
  --full \
  --deterministic \
  --first_frame
```

The released evaluator path is limited to `JointsFull`, `JointsBody`, and `JointsHands`, all without normalization.

## Citation

If you find our work helpful, please cite:

```bibtex
@misc{ruizponce2025interact2arfullbodyhumanhumaninteraction,
  title={Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models},
  author={Pablo Ruiz-Ponce and Sergio Escalera and José García-Rodríguez and Jiankang Deng and Rolandos Alexandros Potamias},
  year={2025},
  eprint={2512.19692},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.19692},
}
```

## License

This repository is released under the software license agreement in [LICENSE](LICENSE).
