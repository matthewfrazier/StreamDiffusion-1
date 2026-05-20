import asyncio
import io
import os
import random
import sys
import time

import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading SD-turbo + TAESD...")
wrapper = StreamDiffusionWrapper(
    model_id_or_path="stabilityai/sd-turbo",
    t_index_list=[0],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    acceleration="none",
    mode="txt2img",
    use_lcm_lora=False,
    use_denoising_batch=True,
    cfg_type="none",
    seed=42,
    use_safety_checker=False,
    dtype=torch.float16,
    device="cuda",
)

wrapper.prepare(prompt="warmup", num_inference_steps=50, guidance_scale=1.0)
for _ in range(5):
    wrapper.stream.txt2img_sd_turbo(1)
torch.cuda.synchronize()
print("Model ready.")

app = FastAPI()
gpu_lock = asyncio.Lock()


@app.get("/generate")
async def generate(
    prompt: str = Query("a photo of a cat"),
    seed: int = Query(-1),
    num_inference_steps: int = Query(50),
    guidance_scale: float = Query(1.0),
):
    prompt = prompt.strip()
    if not prompt or len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt must be 1-1000 characters")
    if num_inference_steps < 1 or num_inference_steps > 50:
        raise HTTPException(status_code=400, detail="Steps must be 1-50")
    if guidance_scale < 0 or guidance_scale > 20:
        raise HTTPException(status_code=400, detail="Guidance scale must be 0-20")
    if seed < -1 or seed > 2**32 - 1:
        raise HTTPException(status_code=400, detail="Seed must be -1 to 4294967295")

    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    try:
        async with gpu_lock:
            wrapper.stream.prepare(
                prompt,
                "",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.manual_seed(actual_seed),
                seed=actual_seed,
            )

            torch.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)

            start = time.perf_counter()
            image_tensor = wrapper.stream.txt2img_sd_turbo(1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            image = postprocess_image(image_tensor.cpu(), output_type="pil")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Generation-Time": f"{elapsed:.4f}",
            "X-Seed": str(actual_seed),
            "X-Steps": str(num_inference_steps),
            "X-Guidance-Scale": str(guidance_scale),
            "X-Width": "512",
            "X-Height": "512",
            "Access-Control-Expose-Headers": "X-Generation-Time, X-Seed, X-Steps, X-Guidance-Scale, X-Width, X-Height",
            "Cache-Control": "no-store",
        },
    )


PAGE_HTML = """<!DOCTYPE html>
<html lang="en" class="sl-theme-dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StreamDiffusion</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/themes/dark.css">
<script type="module" src="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/shoelace-autoloader.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sl-font-sans);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-900);min-height:100vh;padding:16px}
.layout{display:grid;grid-template-columns:260px 1fr;gap:16px;max-width:1100px;margin:0 auto}
@media(max-width:768px){.layout{grid-template-columns:1fr}}
.sidebar{display:flex;flex-direction:column;gap:12px}
.main{display:flex;flex-direction:column;gap:12px}
.prompt-row{display:flex;gap:8px;align-items:flex-end}
.prompt-row sl-textarea{flex:1}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{touch-action:manipulation;-webkit-tap-highlight-color:transparent;border:1px solid var(--sl-color-neutral-300);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-700);padding:4px 14px;border-radius:9999px;font:inherit;font-size:var(--sl-font-size-small);cursor:pointer;user-select:none;transition:background .15s,color .15s,border-color .15s}
.chip[data-active="true"]{background:var(--sl-color-primary-600);color:#fff;border-color:var(--sl-color-primary-600)}
.chip:focus-visible{outline:2px solid var(--sl-color-primary-600);outline-offset:2px}
.result{position:relative;min-height:200px;display:flex;align-items:center;justify-content:center;background:var(--sl-color-neutral-50);border-radius:var(--sl-border-radius-large);overflow:hidden}
.result img{max-width:100%;display:block}
.result.loading img{opacity:.3}
.result.loading sl-spinner{display:inline-flex}
.result sl-spinner{display:none}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:var(--sl-font-size-small);font-family:var(--sl-font-mono)}
.meta-grid .lbl{color:var(--sl-color-neutral-500)}
.meta-grid .val{color:var(--sl-color-primary-500)}
.share-row{display:flex;gap:8px;align-items:center}
.share-row sl-input{flex:1}
.seed-row{display:flex;gap:4px;align-items:flex-end}
.params-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
sl-details::part(header){font-size:var(--sl-font-size-small);padding:8px 12px}
sl-details::part(content){padding:8px 12px}
.source-item{display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--sl-color-neutral-100)}
.source-item sl-input{flex:1}
.source-list{display:flex;flex-direction:column;gap:0}
.sidebar-heading{font-size:var(--sl-font-size-small);font-weight:600;color:var(--sl-color-neutral-500);text-transform:uppercase;letter-spacing:0.05em;padding:4px 0}
.muted{font-size:var(--sl-font-size-small);color:var(--sl-color-neutral-500)}
</style>
</head>
<body>
<div class="layout">

  <aside class="sidebar">
    <sl-card>
      <div slot="header"><strong>Sources</strong></div>
      <div class="source-list" id="sourceList">
        <p class="muted">Text sources stream into the prompt for continuous generation.</p>
      </div>
      <div style="margin-top:8px;display:flex;gap:4px">
        <sl-input id="newSourceUrl" placeholder="https://..." size="small" style="flex:1"></sl-input>
        <sl-icon-button name="plus-lg" label="Add source" id="addSourceBtn"></sl-icon-button>
      </div>
    </sl-card>

    <sl-card>
      <div slot="header">Generation Params</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <sl-input id="seed" label="Seed (-1 = random)" type="number" value="-1" size="small"></sl-input>
        <div class="seed-row">
          <sl-icon-button name="dash-lg" label="Previous seed" id="seedMinus"></sl-icon-button>
          <sl-icon-button name="plus-lg" label="Next seed" id="seedPlus"></sl-icon-button>
          <sl-icon-button name="arrow-clockwise" label="Regenerate" id="seedRegen"></sl-icon-button>
        </div>
        <sl-input id="guidance" label="Guidance Scale" type="number" value="1.0" step="0.1" min="0" size="small"></sl-input>
      </div>
    </sl-card>
  </aside>

  <div class="main">

    <sl-card>
      <div slot="header"><strong>StreamDiffusion</strong> &mdash; RTX 5070</div>
      <div class="prompt-row">
        <sl-textarea id="prompt" label="Prompt" rows="3" value="a beautiful landscape, mountains, sunset, photorealistic" resize="auto"></sl-textarea>
        <sl-button variant="primary" size="large" id="genBtn">Generate</sl-button>
      </div>
    </sl-card>

    <sl-details summary="Style, Composition, Lighting" open>
      <sl-details summary="Style" open>
        <div class="chips" id="styleChips">
          <button class="chip" data-v="photorealistic, 8K, detailed">Photorealistic</button>
          <button class="chip" data-v="oil painting, impressionist, brushstrokes">Oil Painting</button>
          <button class="chip" data-v="watercolor painting, soft edges, wet media">Watercolor</button>
          <button class="chip" data-v="digital art, concept art, trending on artstation">Digital Art</button>
          <button class="chip" data-v="pencil sketch, graphite, hand-drawn">Pencil Sketch</button>
          <button class="chip" data-v="anime style, cel shaded, vibrant colors">Anime</button>
          <button class="chip" data-v="pixel art, 16-bit, retro game">Pixel Art</button>
          <button class="chip" data-v="3D render, octane render, volumetric lighting">3D Render</button>
          <button class="chip" data-v="cinematic, anamorphic lens, film grain, color graded">Cinematic</button>
          <button class="chip" data-v="minimalist, clean lines, flat design">Minimalist</button>
        </div>
      </sl-details>
      <sl-details summary="Composition">
        <div class="chips" id="compChips">
          <button class="chip" data-v="close-up, macro, tight framing">Close-up</button>
          <button class="chip" data-v="wide angle, expansive, establishing shot">Wide Angle</button>
          <button class="chip" data-v="bird's eye view, top-down, aerial">Bird's Eye</button>
          <button class="chip" data-v="low angle, worm's eye view, looking up">Low Angle</button>
          <button class="chip" data-v="portrait, centered subject, shallow depth of field">Portrait</button>
          <button class="chip" data-v="panoramic, ultrawide, landscape orientation">Panoramic</button>
          <button class="chip" data-v="symmetrical, centered composition, balanced">Symmetrical</button>
          <button class="chip" data-v="rule of thirds, off-center subject">Rule of Thirds</button>
          <button class="chip" data-v="bokeh background, blurred background, subject isolation">Bokeh</button>
          <button class="chip" data-v="tilt-shift, miniature effect, selective focus">Tilt-Shift</button>
        </div>
      </sl-details>
      <sl-details summary="Lighting">
        <div class="chips" id="lightChips">
          <button class="chip" data-v="golden hour, warm sunlight, long shadows">Golden Hour</button>
          <button class="chip" data-v="dramatic lighting, chiaroscuro, high contrast">Dramatic</button>
          <button class="chip" data-v="soft diffused light, overcast, even lighting">Soft/Diffused</button>
          <button class="chip" data-v="neon lights, cyberpunk, glowing">Neon</button>
          <button class="chip" data-v="moonlight, nighttime, cool blue tones">Moonlit</button>
          <button class="chip" data-v="studio lighting, rim light, professional">Studio</button>
          <button class="chip" data-v="backlit, silhouette, halo light">Backlit</button>
        </div>
      </sl-details>
    </sl-details>

    <sl-details summary="Advanced (requires model changes)">
      <div style="display:flex;flex-direction:column;gap:12px">
        <div>
          <div class="sidebar-heading">Negative Prompt <sl-badge variant="danger" pill>needs cfg mode</sl-badge></div>
          <sl-textarea id="negPrompt" placeholder="blurry, low quality, distorted, watermark..." rows="2" disabled size="small"></sl-textarea>
        </div>
        <div>
          <div class="sidebar-heading">Image-to-Image <sl-badge variant="danger" pill>needs model reload</sl-badge></div>
          <p class="muted">Redraw an uploaded image guided by the prompt. Requires reloading in img2img mode (~15s).</p>
        </div>
        <div>
          <div class="sidebar-heading">ControlNet <sl-badge variant="danger" pill>needs download</sl-badge></div>
          <sl-select placeholder="Select type" disabled size="small"><sl-option value="canny">Canny Edge</sl-option><sl-option value="depth">Depth Map</sl-option><sl-option value="pose">OpenPose</sl-option><sl-option value="scribble">Scribble</sl-option></sl-select>
        </div>
        <div>
          <div class="sidebar-heading">LoRA Adapters <sl-badge variant="danger" pill>needs download</sl-badge></div>
          <sl-select placeholder="None loaded" disabled size="small"></sl-select>
        </div>
        <div>
          <div class="sidebar-heading">IP-Adapter <sl-badge variant="danger" pill>needs download</sl-badge></div>
          <p class="muted">Feed a reference image to transfer its style/content. Requires IP-Adapter weights (~100 MB).</p>
        </div>
      </div>
    </sl-details>

    <div class="result" id="resultBox">
      <sl-spinner style="font-size:2rem"></sl-spinner>
      <span id="placeholder" style="color:var(--sl-color-neutral-400)">Image will appear here</span>
      <img id="resultImg" style="display:none" alt="Generated image">
    </div>

    <sl-card id="metaBox" style="display:none">
      <div slot="header">Generation Details</div>
      <div class="meta-grid">
        <span class="lbl">Time</span><span class="val" id="mTime"></span>
        <span class="lbl">Seed</span><span class="val" id="mSeed"></span>
        <span class="lbl">Denoise</span><span class="val">1-step (SD-turbo)</span>
        <span class="lbl">Guidance</span><span class="val" id="mGuidance"></span>
        <span class="lbl">Size</span><span class="val">512x512</span>
        <span class="lbl">Model</span><span class="val">sd-turbo + taesd</span>
        <span class="lbl">Modifiers</span><span class="val" id="mMods"></span>
      </div>
    </sl-card>

    <div class="share-row" id="shareBox" style="display:none">
      <sl-input id="shareUrl" readonly size="small" style="flex:1"></sl-input>
      <sl-button size="small" id="copyBtn">Copy URL</sl-button>
    </div>

  </div>
</div>
<script type="module">
function getActiveModifiers() {
  const mods = [];
  document.querySelectorAll('.chip[data-active="true"]').forEach(c => mods.push(c.dataset.v));
  return mods;
}

function buildFullPrompt() {
  const base = document.getElementById('prompt').value.trim();
  const mods = getActiveModifiers();
  if (!mods.length) return base;
  return base + ', ' + mods.join(', ');
}

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chip.dataset.active = chip.dataset.active === 'true' ? 'false' : 'true';
  });
});

async function generate() {
  const btn = document.getElementById('genBtn');
  const box = document.getElementById('resultBox');
  const img = document.getElementById('resultImg');
  const ph = document.getElementById('placeholder');

  const fullPrompt = buildFullPrompt();
  const seed = document.getElementById('seed').value;
  const guidance = document.getElementById('guidance').value;

  btn.loading = true;
  box.classList.add('loading');
  if (ph) ph.style.display = 'none';

  const params = new URLSearchParams({
    prompt: fullPrompt, seed, num_inference_steps: '50', guidance_scale: guidance
  });

  try {
    const resp = await fetch('/generate?' + params.toString());
    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      throw new Error(err?.detail || 'Generation failed: ' + resp.status);
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    img.onload = () => URL.revokeObjectURL(img._prev);
    img._prev = url;
    img.src = url;
    img.style.display = 'block';

    const genTime = resp.headers.get('X-Generation-Time');
    const actualSeed = resp.headers.get('X-Seed');
    const respGuidance = resp.headers.get('X-Guidance-Scale');

    document.getElementById('mTime').textContent = parseFloat(genTime).toFixed(3) + 's';
    document.getElementById('mSeed').textContent = actualSeed;
    document.getElementById('mGuidance').textContent = respGuidance;
    const modCount = document.querySelectorAll('.chip[data-active="true"]').length;
    document.getElementById('mMods').textContent = modCount ? modCount + ' active' : 'none';
    document.getElementById('metaBox').style.display = '';

    const shareParams = new URLSearchParams({
      prompt: fullPrompt, seed: actualSeed, guidance_scale: respGuidance
    });
    const shareUrl = window.location.origin + '/?' + shareParams.toString();
    document.getElementById('shareUrl').value = shareUrl;
    document.getElementById('shareBox').style.display = '';

    if (seed === '-1') document.getElementById('seed').value = actualSeed;
  } catch (e) {
    const dlg = Object.assign(document.createElement('sl-alert'), {
      variant: 'danger', closable: true, duration: 5000,
      innerHTML: '<sl-icon slot="icon" name="exclamation-triangle"></sl-icon>' + e.message
    });
    document.body.append(dlg);
    dlg.toast();
  } finally {
    btn.loading = false;
    box.classList.remove('loading');
  }
}

document.getElementById('genBtn').addEventListener('click', generate);
document.getElementById('seedMinus').addEventListener('click', () => {
  const el = document.getElementById('seed');
  const v = parseInt(el.value, 10);
  if (isNaN(v) || v < 0) return;
  el.value = Math.max(0, v - 1);
  generate();
});
document.getElementById('seedPlus').addEventListener('click', () => {
  const el = document.getElementById('seed');
  const v = parseInt(el.value, 10);
  if (isNaN(v) || v < 0) return;
  el.value = v + 1;
  generate();
});
document.getElementById('seedRegen').addEventListener('click', generate);
document.getElementById('copyBtn').addEventListener('click', () => {
  const el = document.getElementById('shareUrl');
  navigator.clipboard.writeText(el.value).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy URL', 1500);
  });
});

window.addEventListener('DOMContentLoaded', () => {
  const p = new URLSearchParams(window.location.search);
  if (p.has('prompt')) {
    document.getElementById('prompt').value = p.get('prompt');
    if (p.has('seed')) document.getElementById('seed').value = p.get('seed');
    if (p.has('guidance_scale')) document.getElementById('guidance').value = p.get('guidance_scale');
    generate();
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8384)
