"""Illumination-adaptive low-light restoration - Gradio app (Problem Statement P1).

Run locally:  python app.py          (http://localhost:7860)
Docker / HF Spaces: see README.md
"""
import os
import time
import numpy as np
import pandas as pd
import cv2
import gradio as gr

from lowlight.pipeline import LowLightPipeline
from lowlight.metrics import evaluate_all

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPE = LowLightPipeline(os.path.join(ROOT, 'ckpts'), threads=os.cpu_count())
MODES = ['Hybrid (recommended)', 'Classical only', 'Deep learning only', 'Compare all three']
MAX_UPLOAD_SIDE = 4096          # hard guard; processing size is set by the slider


# ----------------------------------------------------------------------------- helpers
def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def colorize(m, cmap=cv2.COLORMAP_MAGMA):
    m = np.asarray(m, np.float32)
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    m = np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap(to_u8(m), cmap), cv2.COLOR_BGR2RGB)


def stage_gallery(res):
    """Intermediate visualisations (illumination, gain, reflectance, candidates, fusion weights ...)."""
    s = res.stages
    items = [(to_u8(s['input']), 'Input')]
    spec = [('L_classical', 'Classical illumination (guided max-RGB)', 'map'),
            ('I_delta_dl', 'Learned illumination (RetinexNet, relit)', 'map'),
            ('L_fused', 'Fused illumination (log-domain, edge-refined)', 'map'),
            ('gain_map_fused', 'Gain map  (L_out / L_in)', 'map'),
            ('noise_strength', 'Noise-aware denoise strength', 'map'),
            ('R_fused', 'Reflectance (fused + denoised)', 'img'),
            ('S_dl_raw', 'Raw RetinexNet output', 'img'),
            ('cand_classical', 'Candidate: classical', 'img'),
            ('cand_dl', 'Candidate: DL + post-processing', 'img'),
            ('cand_hybrid', 'Candidate: illumination-fused', 'img'),
            ('fusion_weights', 'IQA fusion weights  (R=classical, G=DL, B=fused)', 'img'),
            ('L_classical_adj', 'Classical illumination after adaptive gamma', 'map')]
    for key, label, kind in spec:
        if key in s:
            v = s[key]
            items.append((colorize(v) if kind == 'map' else to_u8(v), label))
    items.append((to_u8(s['output']), 'Output'))
    return items


def metrics_table(results, inp, gt):
    rows = []
    for name, r in results.items():
        m = evaluate_all(r.output, inp, gt, with_slow=False)
        row = {'Method': name, 'Latency (ms)': round(r.timings['total'] * 1000)}
        if gt is not None:
            row.update({'PSNR (dB)': round(m['PSNR'], 2), 'SSIM': round(m['SSIM'], 3), 'ΔE00': round(m['DeltaE00'], 2)})
        row.update({'Naturalness proxy ↓': round(m['NatProxy'], 2), 'Noise ratio ↓': round(m['NoiseRatio'], 3),
                    'Edge preservation ↑': round(m['EdgePres'], 3), 'New overexposure %': round(m['Overexp%'], 2),
                    'Hue shift ° ↓': round(m['HueShift'], 1), 'Mean luma': round(m['MeanLuma'], 2)})
        rows.append(row)
    return pd.DataFrame(rows)


def diagnosis_md(res, notes):
    s, p = res.stats, res.params
    lines = [f"**Scene diagnosis** - lighting level: **{s['level']}** · log-mean luma {s['log_mean_luma']:.3f} · "
             f"noise σ {s['noise_sigma']:.4f} · lighting non-uniformity {s['nonuniformity']:.2f} · colour cast {s['cast']:.1f} · "
             f"{s['size'][1]}×{s['size'][0]} px",
             f"**Adaptive parameters chosen** - exposure target {p['target_illum']:.2f} · DL weight {p['w_dl']:.2f} · "
             f"CLAHE clip {p['clahe_clip']:.2f} (blend {p['clahe_alpha']:.2f}) · denoise ×{p['dn_scale']:.2f} · "
             f"enhancement strength {p['enh_strength']:.2f}"]
    t = res.timings
    lines.append('**Timing** - ' + ' · '.join(f'{k.replace("_", " ")} {v * 1000:.0f} ms' for k, v in t.items()))
    lines += [f'ℹ️ {n}' for n in notes]
    return '\n\n'.join(lines)


# ----------------------------------------------------------------------------- main callback
def enhance(image, mode, max_side, auto_exposure, exposure, gt_image, progress=gr.Progress()):
    if image is None:
        raise gr.Error('Please upload an image (or click an example) first.')
    h, w = image.shape[:2]
    if min(h, w) < 32:
        raise gr.Error('Image is too small (min 32 px per side).')
    if max(h, w) > MAX_UPLOAD_SIDE:
        raise gr.Error(f'Image is too large (max {MAX_UPLOAD_SIDE}px per side).')
    progress(0.05, desc='Analysing scene...')
    notes = []
    target = None if auto_exposure else float(exposure)
    pre = PIPE._prep(image, int(max_side))
    if pre.mean() < 0.004:
        notes.append('This image is almost completely black - there is essentially no signal to recover; '
                     'expect mostly amplified noise.')
    if int(max_side) < max(h, w):
        notes.append(f'Image downscaled from {w}×{h} to {pre.shape[1]}×{pre.shape[0]} for speed (see "Max processing size").')
    gt = None
    if gt_image is not None:
        g = PIPE._prep(gt_image, None)
        if g.shape[:2] != pre.shape[:2]:
            g = cv2.resize(g, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)
        gt = g

    results = {}
    if mode.startswith('Compare'):
        progress(0.2, desc='Classical pipeline...')
        results['Classical'] = PIPE.process(pre, 'classical', manual_target=target)
        if PIPE.dl_available:
            progress(0.4, desc='Deep learning (RetinexNet)...')
            results['Deep learning'] = PIPE.process(pre, 'dl')
            progress(0.7, desc='Hybrid fusion...')
            results['Hybrid'] = PIPE.process(pre, 'hybrid', manual_target=target)
        main = results.get('Hybrid', results['Classical'])
    else:
        key = 'classical' if mode.startswith('Classical') else 'dl' if mode.startswith('Deep') else 'hybrid'
        progress(0.3, desc={'classical': 'Classical pipeline...', 'dl': 'Deep learning...',
                            'hybrid': 'Hybrid pipeline (classical + RetinexNet + fusion)...'}[key])
        r = PIPE.process(pre, key, manual_target=target)
        name = {'classical': 'Classical', 'dl': 'Deep learning', 'hybrid': 'Hybrid'}[key]
        if any('falling back' in n for n in r.notes):
            name = 'Classical (fallback)'
        results[name] = r
        main = r
    notes += main.notes
    progress(0.9, desc='Computing metrics...')
    table = metrics_table(results, pre, gt)
    gallery = stage_gallery(main) if len(main.stages) > 2 else [(to_u8(pre), 'Input'), (to_u8(main.output), 'Output')]
    if mode.startswith('Compare'):
        gallery = [(to_u8(pre), 'Input')] + [(to_u8(r.output), n) for n, r in results.items()] + \
                  ([(to_u8(gt), 'Ground truth')] if gt is not None else []) + gallery[1:-1]
    out = to_u8(main.output)
    return (to_u8(pre), out), out, gallery, table, diagnosis_md(main, notes)


CSS = """
.gradio-container {max-width: 1280px !important}
#title h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0}
#title p {opacity: .75; margin-top: 4px}
"""

THEME = gr.themes.Soft(primary_hue='amber', secondary_hue='orange', neutral_hue='slate')


def read_doc(name, fallback):
    p = os.path.join(ROOT, 'docs', name)
    return open(p, encoding='utf-8').read() if os.path.exists(p) else fallback


def build():
    examples = sorted(os.path.join(ROOT, 'samples', f) for f in os.listdir(os.path.join(ROOT, 'samples'))
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')))
    with gr.Blocks(title='Low-light restoration: classical × deep × hybrid') as demo:
        gr.Markdown('# 🌙 Adaptive Low-Light Restoration\nIllumination-adaptive hybrid of classical Retinex/guided-filter '
                    'processing and RetinexNet, fused by a no-reference image-quality strategy.', elem_id='title')
        if not PIPE.dl_available:
            gr.Markdown(f'⚠️ **Deep-learning weights not found** ({PIPE.error}). Running in classical-only mode.')
        with gr.Tab('Enhance'):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label='Low-light image', type='numpy', image_mode='RGB', sources=['upload', 'clipboard'])
                    mode = gr.Radio(MODES, value=MODES[0], label='Approach')
                    with gr.Accordion('Options', open=False):
                        max_side = gr.Slider(256, 1280, value=640, step=64, label='Max processing size (px)',
                                             info='Larger = more detail, slower on CPU')
                        auto = gr.Checkbox(True, label='Auto exposure (adaptive)')
                        expo = gr.Slider(0.35, 0.8, value=0.55, step=0.01, label='Manual exposure target',
                                         info='Used only when auto exposure is off')
                        gt = gr.Image(label='Optional ground-truth (enables PSNR / SSIM / ΔE)', type='numpy', image_mode='RGB')
                    btn = gr.Button('Enhance', variant='primary')
                with gr.Column(scale=2):
                    slider = gr.ImageSlider(label='Before ⟷ After', type='numpy') \
                        if hasattr(gr, 'ImageSlider') else gr.Image(visible=False)
                    diag = gr.Markdown()
            out_img = gr.Image(label='Enhanced output (download)', type='numpy', visible=True)
            table = gr.Dataframe(label='Metrics (no-reference; full-reference when ground truth is supplied)', wrap=True)
            gal = gr.Gallery(label='Intermediate visualisations', columns=4, height='auto', object_fit='contain')
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=[inp], label='Examples (LOL test images)')
            btn.click(enhance, [inp, mode, max_side, auto, expo, gt], [slider, out_img, gal, table, diag])
        with gr.Tab('How it works'):
            gr.Markdown(read_doc('DESIGN.md', 'See docs/DESIGN.md'))
        with gr.Tab('Known failure cases'):
            gr.Markdown(read_doc('FAILURE_ANALYSIS.md', 'See docs/FAILURE_ANALYSIS.md'))
    demo.queue(max_size=10, default_concurrency_limit=2)
    return demo


demo = build()

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=int(os.environ.get('PORT', 7860)), theme=THEME, css=CSS,
                show_error=True)
