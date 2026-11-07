# 🌙 Adaptive Low-Light Restoration: Classical × RetinexNet × Hybrid

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://lowlight-enhancer-cv.netlify.app)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **SoCSE AI Day 2026 — Problem Statement 1 (NexGen Creators)**  
> **Team:** Abubakar Khan (`1RUA24CSE0010`) · Abhishek D Nagoor (`1RUA24CSE0009`) · Aditi M (`1RUA24CSE0019`)  
> **Institution:** School of Computer Science & Engineering, RV University

---

## 🌟 Overview

Low-light photography faces a severe fundamental trade-off: dim images are underexposed, heavily corrupted by sensor noise, color-distorted, and non-uniformly illuminated. Standard fixed-gamma enhancement amplifies noise into harsh grain and blows out existing light sources.

This project implements an **illumination-adaptive low-light restoration framework** that synergistically combines:
1. **Classical Computer Vision:** Guided-filter Retinex decomposition, adaptive gamma mapping, noise-aware guided denoising, CLAHE, and soft-knee highlight roll-off.
2. **Deep Learning (RetinexNet):** Pretrained neural DecomNet and RelightNet architectures modified to run in real-time on CPU/GPU.
3. **No-Reference Quality Hybrid Fusion:** Dynamic multi-candidate fusion in the log-illumination domain weighted by Mertens-style perceptual quality metrics (well-exposedness, edge contrast, saturation, and noise penalties).

---

## 🌐 Public Live Demo

Experience the interactive restoration engine live in your browser:  
👉 **[https://lowlight-enhancer-cv.netlify.app](https://lowlight-enhancer-cv.netlify.app)**

* **Interactive Before ⟷ After Slider:** Real-time drag comparison.
* **Preloaded Test Samples:** Instant one-click evaluations on dark scenes (`Lumen`, `Orbit`, `Strada`, `Ridge`).
* **Zero-Latency In-Browser Processing:** Fast client-side Canvas and WebGL execution with 0% server delay.
* **14 Intermediate Visualizations:** Thermal Magma heatmaps for illumination, gain, noise, and quality weights.

---

## 📊 Visual Benchmarks & Experimental Results

### 1. Extreme Low-Light Scene Restoration
![Figure 3.2(a) - Extreme Low-Light](figures/Figure_3_2_a_Extreme_LowLight.png)
* **Figure 3.2(a):** Severe under-exposure restoration on dark vehicle bodywork. The hybrid engine lifts log-mean luminance from $0.048 \to 0.521$, achieves **$18.92\text{ dB}$ PSNR** and **$0.564$ SSIM** with **$0\%$ newly clipped highlights**.

---

### 2. Backlit & Non-Uniform Scene Comparison (4-Way Ablation)
![Figure 3.2(b) - Non-Uniform Comparison](figures/Figure_3_2_b_NonUniform_Comparison.png)
* **Figure 3.2(b):** Side-by-side comparison across all branches:
  * *(a) Original Input:* High dynamic range with deep shadows and glare.
  * *(b) Classical Retinex:* Sharp edge preservation ($0.896$), but slight grain in deep shadows.
  * *(c) RetinexNet (DL):* Smooth relighting, but exhibits characteristic warm orange cast and $6.56\%$ highlight clipping.
  * *(d) Hybrid Fusion (Ours):* Natural color fidelity, clean noise suppression, and zero clipped highlights.

---

### 3. Pipeline Intermediate Diagnostic States
![Figure 3.2(c) - Intermediate States](figures/Figure_3_2_c_Intermediate_States.png)
* **Figure 3.2(c):** Internal scientific diagnostics:
  1. **Classical Illumination ($L_{adj}$):** Edge-preserving guided max-RGB illumination.
  2. **Learned Illumination ($\Delta I$):** RetinexNet U-Net relighting map.
  3. **Fused Illumination ($L_{fused}$):** Log-domain illumination blend refined by luma guidance.
  4. **Denoised Reflectance ($R$):** Gain-proportional noise suppression on intrinsic reflectance.
  5. **Illumination Gain Map ($L_{out}/L_{in}$):** Spatial amplification field.
  6. **IQA Merge Weights:** Per-pixel Mertens quality map ($\text{Red}=\text{Classical}, \text{Green}=\text{DL}, \text{Blue}=\text{Fused}$).

---

## 🔬 Quantitative Comparison

| Metric | Classical (Guided) | RetinexNet (Deep Learning) | Hybrid Pipeline (Ours) |
| :--- | :---: | :---: | :---: |
| **Noise Ratio** ↓ *(lower is better)* | `0.021` | `0.030` | **`0.018`** *(Lowest)* |
| **Edge Preservation** ↑ *(higher is better)* | `0.896` | `0.865` | **`0.921`** *(Highest)* |
| **Newly Clipped Pixels** *(overexposure %)* | `0.00%` | `6.56%` | **`0.00%`** *(Safe)* |
| **Naturalness Proxy** ↓ | `1.42` | `1.68` | **`1.38`** *(Best)* |
| **Hue Shift** ↓ | `3.1°` | `8.4°` | **`3.8°`** |
| **Synthetic PSNR / SSIM** | $17.14\text{ dB} / 0.452$ | $15.04\text{ dB} / 0.410$ | **$16.46\text{ dB} / 0.530$** |
| **Latency (CPU @ 640px)** | $\approx 0.13\text{ s}$ | $\approx 3.30\text{ s}$ | $\approx 3.80\text{ s}$ |

---

## 🏗️ System Architecture

```
                             ┌───────────────────────┐
                             │  Input Image S(x, y)  │
                             └──────────┬────────────┘
                                        │
                             ┌──────────▼────────────┐
                             │    Scene Diagnosis    │
                             │ (Luma, Noise σ, Cast) │
                             └──────────┬────────────┘
                                        │
                             ┌──────────▼────────────┐
                             │  Adaptive Parameters  │
                             │ (Target L, γ, CLAHE)  │
                             └────┬──────┬───────┬───┘
                                  │      │       │
             ┌────────────────────┘      │       └───────────────────┐
             │                           │                           │
┌────────────▼───────────────┐ ┌─────────▼───────────────┐ ┌─────────▼───────────────┐
│     Classical Branch       │ │   Deep Learning Branch  │ │   Illumination Fusion   │
│ • Guided Max-RGB (L)       │ │ • DecomNet (R, I)       │ │ • Log-domain illumination │
│ • Retinex: R = S / L       │ │ • RelightNet (ΔI)       │ │   fusion: (L_adj, ΔI)   │
│ • Adaptive Gamma: L^γ      │ │ • Relit: S_dl = R · ΔI  │ │ • Luma edge refinement   │
│ • Noise-aware denoise on R │ │                         │ │                         │
│ • Candidate C              │ │ • Candidate D           │ │ • Candidate H           │
└────────────┬───────────────┘ └─────────┬───────────────┘ └─────────┬───────────────┘
             │                           │                           │
             └───────────────────┬───────┴───────────────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  No-Reference Quality │
                     │  Pyramid Multi-Fusion │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ Highlight Roll-Off &  │
                     │ Soft-Knee Compression │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ Enhanced Output Image │
                     └───────────────────────┘
```

---

## 📂 Repository Layout

```bash
Cv_image_enhancement-main/
├── app.py                     # Gradio interactive web UI application
├── index.html                 # Production client-side web engine (Netlify)
├── netlify.toml               # Netlify deployment configuration
├── requirements.txt           # Python dependencies (PyTorch, OpenCV, Gradio)
├── Dockerfile                 # Container deployment specification
├── evaluate.py                # Evaluation benchmark & ablation script
├── train.py                   # RetinexNet fine-tuning pipeline
├── DESIGN.md                  # Comprehensive architectural design document
├── FAILURE_ANALYSIS.md        # Edge cases, failure analysis & mitigations
├── lowlight/                  # Core algorithmic engine package
│   ├── __init__.py
│   ├── analysis.py            # Scene statistics (Immerkaer noise, luma, cast)
│   ├── classical.py           # Guided-filter Retinex, CLAHE, adaptive gamma
│   ├── retinexnet.py          # PyTorch DecomNet & RelightNet architectures
│   ├── pipeline.py            # LowLightPipeline orchestrator & fusion
│   ├── metrics.py             # PSNR, SSIM, Delta E, no-reference metrics
│   └── synth.py               # Synthetic test pair generation
├── ckpts/                     # Pretrained model weights directory
├── samples/                   # Preloaded test images (lumen, orbit, strada, ridge)
└── figures/                   # Publication-ready experimental figures (3.2a, 3.2b, 3.2c)
```

---

## 🚀 Quick Start Guide

### Option 1: Local Python Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Cv_image_enhancement.git
   cd Cv_image_enhancement
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # CPU-optimized PyTorch:
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. **Launch the Gradio Web UI:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to: `http://localhost:7860`

---

### Option 2: Run via Docker

```bash
# Build the container
docker build -t lowlight-enhancer .

# Run container on port 7860
docker run -p 7860:7860 lowlight-enhancer
```

---

### Option 3: Reproduce Benchmarks & Ablation

To evaluate the pipeline against real or synthetic data:

```bash
# Smoke test on synthetic image pairs:
python evaluate.py --synthetic --team T01

# Real unseen images (No-reference evaluation):
python evaluate.py --noref_dir samples --team T01

# Full-reference paired evaluation (LOL dataset eval15):
python evaluate.py --data_dir data/eval15 --team T01 --usns "1RUA24CSE0010,1RUA24CSE0009,1RUA24CSE0019"
```

Results and side-by-side figures are automatically generated in `results/`.

---

## ⚠️ Known Failure Modes & Limitations

As documented in [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md):
1. **Near-Zero Signal in Extreme Darkness:** In almost pitch-black regions where SNR approaches 0, noise-aware filters prioritize smoothness over faint textures, resulting in a soft appearance.
2. **Extreme Deep Shadow Fog:** Severe single-gamma boosting can lift deep background shadows into a faint veiling haze.
3. **Pure Deep Learning Relighting:** RetinexNet alone produces a warm orange cast and clips $6.56\%$ of highlight pixels (mitigated by our hybrid branch).

---

## 👥 Contributors

* **Abubakar Khan** (USN: `1RUA24CSE0010`)
* **Abhishek D Nagoor** (USN: `1RUA24CSE0009`)
* **Aditi M** (USN: `1RUA24CSE0019`)

*School of Computer Science and Engineering (SoCSE), RV University, Bengaluru, India.*

---

## 📚 References

1. **Wei, C., Wang, W., Yang, W., & Liu, J.** (2018). *Deep Retinex Decomposition for Low-Light Enhancement*. British Machine Vision Conference (BMVC).
2. **He, K., Sun, J., & Tang, X.** (2013). *Guided Image Filtering*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
3. **Jobson, D. J., Rahman, Z., & Woodell, G. A.** (1997). *A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes*. IEEE Transactions on Image Processing (TIP).
4. **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization (CLAHE)*. Graphics Gems IV.
5. **Mertens, T., Kautz, J., & Van Reeth, F.** (2009). *Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography*. Computer Graphics Forum.
