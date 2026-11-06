---
title: Adaptive Low-Light Restoration
emoji: 🌙
sdk: gradio
app_file: app.py
python_version: "3.12"
---
# Adaptive Low-Light Restoration - Classical × RetinexNet × Hybrid (Hackathon P1)

Illumination-adaptive restoration of dark, noisy, colour-distorted images. Three approaches are implemented and compared:
**classical** (guided-filter Retinex + adaptive gamma + noise-aware denoise + CLAHE + colour/highlight control),
**deep learning** (RetinexNet, your uploaded PyTorch port made CPU-capable), and a **hybrid** that fuses both with a
scene-adaptive, no-reference-quality-aware strategy. See `docs/DESIGN.md`, `docs/architecture.png`, `docs/FAILURE_ANALYSIS.md`.

## Run
```bash
pip install -r requirements.txt     # CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python app.py                        # http://localhost:7860  (set PORT=... to change)
# or
docker build -t lowlight . && docker run -p 7860:7860 lowlight
```
**Deploy (live URL):** create a Hugging Face *Space* (SDK: Gradio, CPU basic) and push this repo (checkpoints are ~2 MB, committed
in `ckpts/`). The front-matter above configures it. *A live URL has not been created for you - that step needs your account.*

## Reproduce the evaluation
```bash
python evaluate.py --data_dir <LOL/eval15>  --team T01 --usns "USN1,USN2,USN3,USN4"   # real paired data (needs low/ + high/)
python evaluate.py --noref_dir samples      --team T01 --no_ablation                   # unseen images, no-reference metrics
python evaluate.py --synthetic              --team T01                                 # smoke test (not real LOL numbers)
```
Writes `results/eval_P1_<TEAM>.md` (first line = USNs) and side-by-side figures. Fill in your USNs/Team ID.
Datasets: LOL, LOL-v2, MIT-Adobe FiveK (links in the hackathon datasheet).

## Train / fine-tune
```bash
python train.py --data_dir data/LOL/our485 --val_dir data/LOL/eval15 --epochs 30 --noise_aug 0.02   # CPU or GPU
```
Fixes vs. the original: real shuffling, no hard-coded CUDA, validation PSNR + best checkpoint, frozen DecomNet in the Relight phase.
**No fine-tuned weights are included** - results use the pretrained 9200-step checkpoint.

## Headline results (reproduced by `results/*.md`)
| Setting | Classical | RetinexNet | Hybrid |
|---|---|---|---|
| Synthetic pairs, PSNR / SSIM (12 imgs, smoke test) | 17.14 / 0.452 | 15.04 / 0.410 | 16.46 / **0.530** |
| Real unseen, noise ratio ↓ / edge pres. ↑ / new over-exp % | 0.021 / 0.896 / 0 | 0.030 / 0.865 / 6.6 | **0.018 / 0.921** / 0 |
| CPU latency @ ≤640 px, 1 core | ~0.1 s | ~3 s | ~4 s |

Honest reading: the hybrid is cleaner and more robust than RetinexNet alone and better on SSIM/noise than classical, but it
does **not** win PSNR against the classical branch in our synthetic test. Official LPIPS/NIQE and real-LOL PSNR/SSIM still need to be run
(see `docs/DESIGN.md` §7 for limitations and ethics).

## Layout
`app.py` UI · `lowlight/` (retinexnet, classical, analysis, pipeline, metrics, synth) · `evaluate.py` · `train.py` · `ckpts/` · `samples/` · `results/` · `docs/`
RetinexNet: Wei, Wang, Yang, Liu, BMVC'18 - cite the paper if you use this code (original README citation retained upstream).
