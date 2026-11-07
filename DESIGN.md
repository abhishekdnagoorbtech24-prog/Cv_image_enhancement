# Design document - Adaptive hybrid low-light restoration (P1)

**Architecture diagram:** `docs/architecture.png` (regenerate with `python docs/make_diagram.py`).

```
image → [1 analysis] → [2 adaptive params] ─┬→ [3 classical branch] ──→ candidate C ─────────────┐
                                           ├→ [4 RetinexNet branch] → candidate D ─────────────┤
                                           └→ [5 illumination fusion (uses 3 & 4)] → cand. H ──┤
                                                                                              ↓
                                          output ← [7 gating + auto-exposure] ← [6 IQA-aware pyramid fusion]
```

## 1. Problem & goals
Restore severely under-exposed, noisy, colour-distorted, non-uniformly lit images while preserving edges/texture and
avoiding over-exposure, halos and unnatural colour; compare classical vs deep vs hybrid; ablate; document failures.

## 2. Classical branch (`lowlight/classical.py`)
| Stage | Method | Why |
|---|---|---|
| Illumination estimation | max-RGB prior refined by a two-scale **guided filter** (self-guided) | edge-preserving, so bright light sources do not bleed (halo control) |
| Decomposition | **Retinex**: R = S / L (per channel), L ≥ S constraint | separates lighting from scene content |
| Exposure | **adaptive gamma** on L, solved so log-mean(L^γ) hits a target | lights with L≈1 barely change → built-in over-exposure protection |
| Denoising | guided filter on reflectance, **pre-smoothed luma guide**, chroma filtered harder than luma, strength ∝ local gain | noise is amplified by the gain, so filter where gain is high; pre-smoothing stops noise being treated as an edge |
| Contrast | **CLAHE** on LAB lightness, blended (α adaptive) | local contrast without global washout |
| Detail | base/detail split, re-inject texture gated by noise map | texture preservation without boosting noise |
| Colour | conservative shades-of-gray white balance (only when a clear cast exists, gains clamped), saturation limiter | avoids neutralising genuine coloured light (neon, sodium lamps) |
| Highlights | hue-preserving soft-knee roll-off | no clipping |
Baseline for comparison: **MSRCR** (multi-scale Retinex with colour restoration).

## 3. Deep-learning branch (`lowlight/retinexnet.py`)
RetinexNet (Wei et al., BMVC'18): DecomNet → (R, I); RelightNet → relit illumination; output S = R·I_delta. Architecture and
parameter names are identical to the supplied PyTorch port, so the pretrained 9200-step LOL checkpoint loads unchanged.
Changes: CPU/GPU agnostic, no `Variable`/`.cuda()`, cached inference option for evaluation. `train.py` fine-tunes from the
checkpoint (lr 1e-4), with optional noise-augmentation of the dark inputs, a real shuffle, a train/val split and best-checkpoint saving.
**Fine-tuning status:** the code is smoke-tested on toy data only; *no fine-tuned weights were produced*, all reported results use
the pretrained checkpoint.

## 4. Hybrid (`lowlight/pipeline.py`)
1. **Scene analysis** (`analysis.py`): log-mean luma → level {extreme, dark, dim, normal}; noise σ (Immerkaer); lighting
   non-uniformity; local contrast; colour cast; clipping ratio.
2. **Adaptive parameters** (`choose_params`): exposure target (lower when real light sources exist), guided-filter eps, denoise
   strength (↑ with noise), CLAHE clip/blend (blend ↓ with noise), detail amount (↓ with noise), DL weight *w* (0.65 extreme → 0.25
   normal), enhancement strength (0 for well-lit inputs). `adaptive=False` gives fixed constants (used in the ablation).
3. **Illumination fusion**: L = L_classical-adjusted^(1-w) · L_RetinexNet^w in the log domain, re-filtered with the input luma as guide;
   reflectance = classical R blended with RetinexNet's chromaticity.
4. **Image-quality-aware fusion**: three candidates (classical C, DL+post D, illumination-fused H). Each pixel gets Mertens-style
   weights (well-exposedness, edge contrast, saturation, **noise penalty**) multiplied by a global no-reference quality score
   (`metrics.nr_quality`: exposure, clipping, noise-to-signal, colour plausibility); candidates are merged with Laplacian
   pyramids to avoid seams/halos.
5. **Gating + auto-exposure**: blend toward the original by `enh_strength`, then bounded global gamma.
Graceful degradation: if the DL weights are missing, the app runs classical-only and says so.

## 5. Evaluation protocol
`evaluate.py` runs MSRCR baseline, classical, RetinexNet-only and hybrid, then an ablation that removes one component at a time.
Metrics: PSNR, SSIM, ΔE00 (needs GT); LPIPS and official NIQE (only if `lpips` / `pyiqa` are installed and weights downloadable);
noise ratio, edge preservation (gradient-map correlation), new over-exposure %, hue shift, detail retention, latency.
Output: `results/eval_P1_<TEAM>.md` (first line = USNs).

## 6. Results (see `results/` for full tables)
**Synthetic pairs (12 images, smoke test only - not LOL):** classical 17.14 dB / 0.452 SSIM, RetinexNet 15.04 / 0.410, hybrid 16.46 / 0.530;
hybrid halves the noise ratio (0.058 vs 0.109 classical). **Real unseen LOL test images (no GT):** hybrid has the lowest noise ratio
(0.018) and highest edge preservation (0.921), zero new over-exposure (RetinexNet alone: 6.6 %).
**Ablation highlights (synthetic):** removing denoising costs −1.4 dB / −0.16 SSIM (largest); removing adaptivity −0.7 dB / −0.10 SSIM;
removing IQA fusion lowers SSIM 0.530 → 0.469 but *raises* PSNR to 16.81; removing the colour stage raises PSNR (16.80) and removing CLAHE raises SSIM (0.546) and lowers noise (but costs 0.1 dB PSNR) on this
noisy data - i.e. these stages buy perceptual contrast/colour at a small full-reference cost. Removing the classical illumination from
the fusion also scored marginally higher (16.65 dB) - the classical term is not clearly justified by this data alone.

## 7. Limitations & ethics (honest statement)
* Numbers on synthetic data are indicative only; real LOL/LOL-v2 numbers must be produced with `evaluate.py --data_dir`.
* The hybrid does **not** beat classical on PSNR in our synthetic test; its advantage is noise/SSIM/colour robustness.
* `NatProxy` is *not* NIQE and is unreliable (it favoured grainy DL output). `HueShift` measured against the *input* penalises
  legitimate colour correction (e.g. removing a cast), so it must not be read as colour error without a ground truth.
* Heuristics (white balance, thresholds in `choose_params`) were set by reasoning and spot-checks, not learned or cross-validated.
* Latency is CPU-bound (~4 s at 640 px, one core); GPU recommended for large images.
* Enhancement can reveal or amplify content that was intentionally dark (privacy/forensics): outputs are *enhancements*, not
  evidence of ground truth, and must not be presented as such. Colour/texture is partly inferred, not recovered.
