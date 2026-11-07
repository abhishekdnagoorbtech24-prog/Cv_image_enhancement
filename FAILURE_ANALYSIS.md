# Known failure cases (observed, with causes)

Figures are in `results/figures/` (Input | MSRCR | Classical | RetinexNet | Hybrid [| GT]).

**1. Extreme darkness + heavy noise → structure lost, sky turns hazy** (`compare_synthetic_rocket_L0p04.jpg`, hybrid PSNR 14.3 dB;
the three rocket images are the lowest-scoring group in our synthetic set, 13.7-14.3 dB). The thin launch-pylon structures are smoothed away and the deep-blue night sky is lifted to a
grey-blue haze (GT is dark and saturated). *Cause:* the max-RGB Retinex model assumes a reflectance whose brightest channel is ≈1,
so dark flat regions get brightened beyond what the scene had; at SNR this low the noise-aware denoiser cannot separate thin edges
from noise and chooses smoothness. *Mitigation ideas:* a floor on gain in near-black regions; denoising with a learned prior.

**2. Extremely dark scene with a colour chart → desaturated, blurry** (`compare_real_unseen_lol_test_3.jpg`). RetinexNet alone gives the most
accurate chart colours but visible grain; the hybrid is smoother but left walls look washed-out and fine texture is soft.
*Cause:* classical Retinex divides by a max-RGB illumination, which discards chroma when the signal is near the noise floor; the fusion
weights favour the low-noise candidates, which are the desaturated ones. *Mitigation:* raise the learned-chroma weight when σ is low.

**3. Dim street scene → veiling haze in deep shadows** (`compare_real_unseen_lol_test_2.jpg`). Shadow areas of buildings/shops
look foggy (low local contrast) after brightening. *Cause:* a single global gamma on the illumination lifts shadows and sky
differently from what local contrast needs; CLAHE blend is deliberately weak on noisy images (to avoid amplifying noise).
*Mitigation:* spatially adaptive contrast in shadows after denoising.

**4. RetinexNet alone: colour cast and over-exposure** (`compare_real_unseen_lol_test_1.jpg`). Orange cast over the night sky and
6.6 % newly clipped pixels on the real test images. The hybrid avoids this, but at the cost of a more conservative look.

**5. Process limitations:** no ground truth for real images here (only no-reference metrics); official NIQE/LPIPS were not computed
(no weights offline); fine-tuning not performed; very large images are downscaled for CPU latency.
