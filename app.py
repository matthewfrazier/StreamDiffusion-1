import asyncio
import functools
import gc
import io
import os
import random
import sys
import time
import uuid

import torch
from fastapi import FastAPI, File, HTTPException, Path, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from typing import Optional
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODELS = {
    "sd-turbo": {
        "id": "stabilityai/sd-turbo",
        "label": "SD-turbo",
        "desc": "1-step distilled, fastest",
        "turbo": True,
        "lcm": False,
        "t_index": [0],
        "guidance": 1.0,
        "loras": [],
    },
    # sdxl-turbo excluded: SDXL UNet requires added_cond_kwargs
    # that StreamDiffusion's txt2img_sd_turbo path doesn't provide
    "dreamshaper-8": {
        "id": "Lykon/dreamshaper-8",
        "label": "DreamShaper 8",
        "desc": "Community all-rounder (SD 1.5)",
        "turbo": False,
        "lcm": True,
        "t_index": [0, 17, 35],
        "guidance": 1.2,
        "loras": [],
    },
    "lcm-dreamshaper": {
        "id": "SimianLuo/LCM_Dreamshaper_v7",
        "label": "LCM DreamShaper v7",
        "desc": "4-step LCM distilled",
        "turbo": False,
        "lcm": True,
        "t_index": [0, 17, 35],
        "guidance": 1.0,
        "loras": [],
    },
}

LORAS = {
    "lcm-sd15": {
        "id": "latent-consistency/lcm-lora-sdv1-5",
        "label": "LCM LoRA (SD 1.5)",
        "desc": "Fast inference on any SD 1.5 model",
        "compat": ["dreamshaper-8"],
    },
    # sdxl-lightning excluded: requires SDXL base which StreamDiffusion doesn't support
}

CONTROLNETS = {
    "canny-sd15": {
        "id": "lllyasviel/control_v11p_sd15_canny",
        "label": "Canny Edge (SD 1.5)",
        "desc": "Edge detection structural control",
        "compat": ["sd-turbo", "dreamshaper-8", "lcm-dreamshaper"],
    },
}

active_model_key = "sd-turbo"
active_controlnet_key = None
wrapper = None
loading_model = False

sources = []


def load_model(model_key, controlnet_key=None):
    global wrapper, active_model_key, active_controlnet_key
    cfg = MODELS[model_key]
    print(f"Loading {cfg['label']}...")

    if wrapper is not None:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()

    lora_dict = None
    for lora_key in cfg["loras"]:
        if lora_key in LORAS:
            lr = LORAS[lora_key]
            path = lr.get("file", lr["id"])
            if lora_dict is None:
                lora_dict = {}
            lora_dict[path] = 1.0

    # Resolve ControlNet model id if requested and compatible
    controlnet_id = None
    if controlnet_key and controlnet_key in CONTROLNETS:
        cn_cfg = CONTROLNETS[controlnet_key]
        if model_key in cn_cfg["compat"]:
            controlnet_id = cn_cfg["id"]
            print(f"Loading ControlNet: {cn_cfg['label']}")
        else:
            print(f"ControlNet {cn_cfg['label']} not compatible with {cfg['label']}, skipping")
            controlnet_key = None

    ip_adapter_path = os.path.join(os.path.dirname(__file__), "models", "ip-adapter")
    has_ip = os.path.exists(os.path.join(ip_adapter_path, "models", "ip-adapter_sd15.bin"))

    wrapper = StreamDiffusionWrapper(
        model_id_or_path=cfg["id"],
        t_index_list=cfg["t_index"],
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=10,
        acceleration="none",
        mode="txt2img",
        use_lcm_lora=cfg["lcm"],
        lora_dict=lora_dict,
        controlnet_id=controlnet_id,
        ip_adapter_path=ip_adapter_path if has_ip else None,
        use_denoising_batch=True,
        cfg_type="full" if not cfg["turbo"] else "none",
        seed=42,
        use_safety_checker=False,
        dtype=torch.float16,
        device="cuda",
    )
    active_controlnet_key = controlnet_key

    wrapper.prepare(prompt="warmup", num_inference_steps=50, guidance_scale=cfg["guidance"])
    warmup_iters = 5 if cfg["turbo"] else wrapper.stream.denoising_steps_num + 2
    for _ in range(warmup_iters):
        if cfg["turbo"]:
            wrapper.stream.txt2img_sd_turbo(1)
        else:
            wrapper.stream.txt2img()
    torch.cuda.synchronize()
    active_model_key = model_key
    print(f"Model ready: {cfg['label']}")


load_model("sd-turbo")

app = FastAPI()
gpu_lock = asyncio.Lock()


@app.get("/models")
async def list_models():
    has_control_image = (
        wrapper is not None
        and wrapper.stream.controlnet is not None
        and wrapper.stream.control_image is not None
    )
    has_ip_adapter = wrapper is not None and wrapper.clip_image_encoder is not None
    has_ip_reference = has_ip_adapter and wrapper.stream.ip_image_embeds is not None
    return JSONResponse({
        "active": active_model_key,
        "loading": loading_model,
        "active_controlnet": active_controlnet_key,
        "has_control_image": has_control_image,
        "has_ip_adapter": has_ip_adapter,
        "has_ip_reference": has_ip_reference,
        "models": {k: {"label": v["label"], "desc": v["desc"], "guidance": v["guidance"]} for k, v in MODELS.items()},
        "loras": {k: {"label": v["label"], "desc": v["desc"], "compat": v["compat"]} for k, v in LORAS.items()},
        "controlnets": {k: {"label": v["label"], "desc": v["desc"], "compat": v["compat"]} for k, v in CONTROLNETS.items()},
    })


class LoadModelRequest(BaseModel):
    model: str
    controlnet: Optional[str] = None


class AddSourceRequest(BaseModel):
    url: str
    label: Optional[str] = None


class SavePromptRequest(BaseModel):
    prompt: str


@app.post("/load_model")
async def switch_model(req: LoadModelRequest):
    global loading_model
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    if req.controlnet and req.controlnet not in CONTROLNETS:
        raise HTTPException(status_code=400, detail=f"Unknown controlnet: {req.controlnet}")
    same_model = req.model == active_model_key
    same_controlnet = req.controlnet == active_controlnet_key
    if same_model and same_controlnet:
        return JSONResponse({"status": "already_loaded", "model": req.model, "guidance": MODELS[req.model]["guidance"]})
    if loading_model:
        raise HTTPException(status_code=409, detail="A model is already loading")
    loading_model = True
    try:
        async with gpu_lock:
            await asyncio.get_event_loop().run_in_executor(
                None, functools.partial(load_model, req.model, controlnet_key=req.controlnet)
            )
        return JSONResponse({
            "status": "loaded",
            "model": req.model,
            "controlnet": active_controlnet_key,
            "guidance": MODELS[req.model]["guidance"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        loading_model = False


@app.get("/generate")
async def generate(
    prompt: str = Query("a photo of a cat"),
    negative_prompt: str = Query(""),
    seed: int = Query(-1),
    num_inference_steps: int = Query(50),
    guidance_scale: float = Query(1.0),
):
    if loading_model:
        raise HTTPException(status_code=503, detail="Model is loading, please wait")
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
    cfg = MODELS[active_model_key]

    try:
        async with gpu_lock:
            wrapper.stream.prepare(
                prompt,
                negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.manual_seed(actual_seed),
                seed=actual_seed,
            )

            torch.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)

            start = time.perf_counter()
            if cfg["turbo"]:
                image_tensor = wrapper.stream.txt2img_sd_turbo(1)
            else:
                for _ in range(wrapper.stream.denoising_steps_num):
                    image_tensor = wrapper.stream.txt2img()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            image = postprocess_image(image_tensor.cpu(), output_type="pil")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.BytesIO()
    image.save(buf, format="PNG")

    neg = negative_prompt.strip()
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Generation-Time": f"{elapsed:.4f}",
            "X-Seed": str(actual_seed),
            "X-Steps": str(num_inference_steps),
            "X-Guidance-Scale": str(guidance_scale),
            "X-Model": active_model_key,
            "X-Model-Label": cfg["label"],
            "X-Cfg-Type": cfg.get("cfg_type", "full" if not cfg["turbo"] else "none"),
            "X-Prompt": prompt[:200],
            "X-Negative-Prompt": neg[:200] if neg else "(none)",
            "X-IP-Adapter": "active" if (wrapper and wrapper.stream.ip_image_embeds is not None) else "off",
            "X-Width": "512",
            "X-Height": "512",
            "Access-Control-Expose-Headers": "X-Generation-Time, X-Seed, X-Steps, X-Guidance-Scale, X-Model, X-Model-Label, X-Cfg-Type, X-Prompt, X-Negative-Prompt, X-IP-Adapter, X-Width, X-Height",
            "Cache-Control": "no-store",
        },
    )


@app.post("/controlnet/upload")
async def controlnet_upload(file: UploadFile = File(...)):
    """Upload an image for ControlNet canny conditioning."""
    if wrapper is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if wrapper.stream.controlnet is None:
        raise HTTPException(
            status_code=400,
            detail="ControlNet is not loaded. Load a model with ControlNet support first.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        async with gpu_lock:
            wrapper.stream.set_controlnet_image(image)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"status": "ok", "message": "ControlNet image set"})


@app.post("/controlnet/clear")
async def controlnet_clear():
    """Clear the ControlNet conditioning image (disable ControlNet guidance)."""
    if wrapper is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    wrapper.stream.clear_controlnet_image()
    return JSONResponse({"status": "ok", "message": "ControlNet image cleared"})


@app.post("/ip-adapter/upload")
async def ip_adapter_upload(file: UploadFile = File(...), scale: float = 0.6):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if wrapper.clip_image_encoder is None:
        raise HTTPException(status_code=400, detail="IP-Adapter not loaded for this model")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embeds = wrapper.encode_ip_image(image)
    wrapper.stream.set_ip_image_embeds(embeds, scale=scale)
    return JSONResponse({"status": "ok", "message": f"IP-Adapter reference set (scale={scale})"})


@app.post("/ip-adapter/clear")
async def ip_adapter_clear():
    if wrapper is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    wrapper.stream.clear_ip_image_embeds()
    return JSONResponse({"status": "ok", "message": "IP-Adapter reference cleared"})


@app.get("/sources")
async def list_sources():
    return JSONResponse(sources)


@app.post("/sources")
async def add_source(req: AddSourceRequest):
    source = {
        "id": str(uuid.uuid4()),
        "url": req.url,
        "label": req.label if req.label else req.url,
    }
    sources.append(source)
    return JSONResponse(source, status_code=201)


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str = Path(...)):
    global sources
    before = len(sources)
    sources = [s for s in sources if s["id"] != source_id]
    if len(sources) == before:
        raise HTTPException(status_code=404, detail="Source not found")
    return JSONResponse({"status": "deleted", "id": source_id})


@app.post("/sources/from-prompt")
async def save_prompt_as_source(req: SavePromptRequest):
    source = {
        "id": str(uuid.uuid4()),
        "url": req.prompt,
        "label": "Saved prompt",
    }
    sources.append(source)
    return JSONResponse(source, status_code=201)


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
@media(max-width:768px){.layout{grid-template-columns:1fr}body{padding:8px}.sidebar{order:2}.main{order:1}.result{min-height:unset}}
.sidebar{display:flex;flex-direction:column;gap:12px}
.main{display:flex;flex-direction:column;gap:12px}
.prompt-row{display:flex;gap:8px;align-items:flex-end}
.prompt-row sl-textarea{flex:1}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{touch-action:manipulation;-webkit-tap-highlight-color:transparent;border:1px solid var(--sl-color-neutral-300);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-700);padding:4px 14px;border-radius:9999px;font:inherit;font-size:var(--sl-font-size-small);cursor:pointer;user-select:none;transition:background .15s,color .15s,border-color .15s}
.chip[data-active="true"]{background:var(--sl-color-primary-600);color:#fff;border-color:var(--sl-color-primary-600)}
.chip:focus-visible{outline:2px solid var(--sl-color-primary-600);outline-offset:2px}
.result{position:relative;min-height:120px;display:flex;align-items:center;justify-content:center;background:var(--sl-color-neutral-50);border-radius:var(--sl-border-radius-large);overflow:hidden}
.result img{max-width:100%;display:block}
.result.loading img{opacity:.3}
.result.loading sl-spinner{display:inline-flex}
.result sl-spinner{display:none}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:var(--sl-font-size-small);font-family:var(--sl-font-mono)}
.meta-grid .lbl{color:var(--sl-color-neutral-500)}
.meta-grid .val{color:var(--sl-color-primary-500)}
.share-row{display:flex;gap:8px;align-items:center}
.share-row sl-input{flex:1}
.controls-row{display:flex;gap:8px;align-items:center;margin-top:8px}
.seed-group{display:flex;align-items:center;gap:2px;flex:1;min-width:0}
.seed-group sl-input{flex:1;min-width:0}
.seed-group sl-icon-button{font-size:var(--sl-font-size-medium)}
sl-details::part(header){font-size:var(--sl-font-size-small);padding:8px 12px}
sl-details::part(content){padding:8px 12px}
.source-item{display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--sl-color-neutral-100)}
.source-item .source-label{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer;border-radius:4px;padding:2px 4px;transition:background .15s}
.source-item .source-label:hover{background:var(--sl-color-neutral-100)}
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
      <div slot="header">Model <sl-badge id="modelStatus" variant="success" pill>ready</sl-badge></div>
      <sl-select id="modelSelect" size="small" value="sd-turbo" hoist>
        <sl-option value="sd-turbo">SD-turbo &mdash; 1-step, fastest</sl-option>
        <sl-option value="dreamshaper-8">DreamShaper 8 &mdash; all-rounder</sl-option>
        <sl-option value="lcm-dreamshaper">LCM DreamShaper &mdash; 4-step LCM</sl-option>
      </sl-select>
      <sl-progress-bar id="modelProgress" style="display:none;margin-top:8px" indeterminate></sl-progress-bar>
      <div style="margin-top:8px">
        <div class="sidebar-heading">LoRAs</div>
        <sl-checkbox id="lora-lcm-sd15" size="small" disabled>LCM LoRA (SD 1.5)</sl-checkbox>
      </div>
    </sl-card>

  </aside>

  <div class="main">

    <div class="result" id="resultBox">
      <sl-spinner style="font-size:2rem"></sl-spinner>
      <span id="placeholder" style="color:var(--sl-color-neutral-400)">Image will appear here</span>
      <img id="resultImg" style="display:none" alt="Generated image">
    </div>

    <sl-card>
      <div class="prompt-row">
        <sl-textarea id="prompt" rows="2" value="a beautiful landscape, mountains, sunset, photorealistic" resize="auto" placeholder="Describe your image..."></sl-textarea>
        <div style="display:flex;flex-direction:column;gap:4px">
          <sl-button variant="primary" size="large" id="genBtn">Generate</sl-button>
          <sl-button size="small" variant="text" id="savePromptBtn">Save</sl-button>
        </div>
      </div>
      <div class="controls-row">
        <div class="seed-group">
          <sl-input id="seed" type="number" value="-1" size="small" placeholder="Seed" no-spin-buttons></sl-input>
          <sl-icon-button name="dash-lg" label="Previous seed" id="seedMinus"></sl-icon-button>
          <sl-icon-button name="plus-lg" label="Next seed" id="seedPlus"></sl-icon-button>
          <sl-icon-button name="arrow-clockwise" label="Regenerate" id="seedRegen"></sl-icon-button>
        </div>
        <sl-input id="guidance" type="number" value="1.0" step="0.1" min="0" size="small" placeholder="CFG" no-spin-buttons style="width:60px"></sl-input>
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
          <div class="sidebar-heading">Negative Prompt <sl-badge variant="success" pill>enabled</sl-badge> <sl-tooltip content="Steers the model away from unwanted qualities. Only effective with CFG-capable models (DreamShaper, LCM) at guidance > 1.0. SD-turbo ignores this."><sl-icon name="info-circle"></sl-icon></sl-tooltip></div>
          <sl-textarea id="negPrompt" placeholder="blurry, low quality, distorted, watermark..." rows="2" size="small"></sl-textarea>
        </div>
        <div>
          <div class="sidebar-heading">Image-to-Image <sl-badge variant="danger" pill>needs model reload</sl-badge> <sl-tooltip content="Redraw an uploaded image guided by your prompt. Controls how much of the original to keep vs. reimagine. Good for style transfer on existing photos."><sl-icon name="info-circle"></sl-icon></sl-tooltip></div>
          <p class="muted">Upload an image and redraw it guided by the prompt. Requires reloading in img2img mode (~15s).</p>
        </div>
        <div>
          <div class="sidebar-heading">ControlNet <sl-badge id="cnBadge" variant="neutral" pill>available</sl-badge> <sl-tooltip content="Controls spatial structure. Upload a reference image and it extracts edges/depth/pose, then forces the output to follow that layout. Great for consistent composition across generations, or turning sketches into rendered images. Does NOT preserve character identity."><sl-icon name="info-circle"></sl-icon></sl-tooltip></div>
          <sl-select id="cnSelect" placeholder="Select type" size="small" value="canny" clearable>
            <sl-option value="canny">Canny Edge (SD 1.5)</sl-option>
            <sl-option value="depth" disabled>Depth Map (coming soon)</sl-option>
            <sl-option value="pose" disabled>OpenPose (coming soon)</sl-option>
            <sl-option value="scribble" disabled>Scribble (coming soon)</sl-option>
          </sl-select>
          <div style="margin-top:8px">
            <label class="muted" style="display:block;margin-bottom:4px">Upload reference image for edge detection:</label>
            <input type="file" id="cnImageInput" accept="image/*" style="font-size:var(--sl-font-size-small)">
          </div>
          <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
            <sl-button id="cnUploadBtn" size="small" variant="primary" disabled>Upload &amp; Apply</sl-button>
            <sl-button id="cnClearBtn" size="small" variant="default">Clear</sl-button>
            <span id="cnStatus" class="muted"></span>
          </div>
        </div>
        <div>
          <div class="sidebar-heading">LoRA Adapters <sl-badge variant="danger" pill>needs download</sl-badge> <sl-tooltip content="Small fine-tuned weight patches that adjust the model's style or add specific concepts. Stack multiple LoRAs to combine effects. Each is 10-200 MB."><sl-icon name="info-circle"></sl-icon></sl-tooltip></div>
          <sl-select placeholder="None loaded" disabled size="small"></sl-select>
        </div>
        <div>
          <div class="sidebar-heading">IP-Adapter <sl-badge id="ipBadge" variant="neutral" pill>available</sl-badge> <sl-tooltip content="Feed a reference face or character image and it transfers that identity into new generations. This is how you get consistent characters across different scenes and prompts."><sl-icon name="info-circle"></sl-icon></sl-tooltip></div>
          <div style="margin-top:4px">
            <label class="muted" style="display:block;margin-bottom:4px">Upload a reference image (face, character, style):</label>
            <input type="file" id="ipImageInput" accept="image/*" style="font-size:var(--sl-font-size-small)">
          </div>
          <div style="margin-top:4px;display:flex;gap:8px;align-items:center">
            <sl-input id="ipScale" type="number" value="0.6" step="0.1" min="0" max="1" size="small" style="width:70px" no-spin-buttons></sl-input>
            <sl-button id="ipUploadBtn" size="small" variant="primary" disabled>Apply Reference</sl-button>
            <sl-button id="ipClearBtn" size="small" variant="default">Clear</sl-button>
          </div>
          <span id="ipStatus" class="muted"></span>
        </div>
      </div>
    </sl-details>

    <sl-card id="metaBox" style="display:none">
      <div slot="header">Generation Details</div>
      <div class="meta-grid">
        <span class="lbl">Prompt</span><span class="val" id="mPrompt" style="word-break:break-word"></span>
        <span class="lbl">Negative</span><span class="val" id="mNeg" style="word-break:break-word"></span>
        <span class="lbl">Model</span><span class="val" id="mModel">sd-turbo</span>
        <span class="lbl">CFG Type</span><span class="val" id="mCfgType"></span>
        <span class="lbl">Guidance</span><span class="val" id="mGuidance"></span>
        <span class="lbl">Steps</span><span class="val" id="mSteps"></span>
        <span class="lbl">Seed</span><span class="val" id="mSeed"></span>
        <span class="lbl">Size</span><span class="val">512x512</span>
        <span class="lbl">Time</span><span class="val" id="mTime"></span>
        <span class="lbl">IP-Adapter</span><span class="val" id="mIpAdapter"></span>
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
const shortcuts = {
  bindings: [
    {key: 'Enter', ctrl: true,  action: 'generate',     desc: 'Generate image'},
    {key: 'Enter', meta: true,  action: 'generate',     desc: 'Generate image'},
    {key: ']',     ctrl: true,  action: 'seedUp',       desc: 'Increment seed'},
    {key: '[',     ctrl: true,  action: 'seedDown',     desc: 'Decrement seed'},
    {key: 'r',     ctrl: true,  shift: true, action: 'regenerate', desc: 'Regenerate (same params)'},
    {key: 'c',     ctrl: true,  shift: true, action: 'copyUrl',   desc: 'Copy share URL'},
    {key: '/',     ctrl: false, action: 'focusPrompt',  desc: 'Focus prompt'},
    {key: 's',     ctrl: false, action: 'focusSeed',    desc: 'Focus seed input'},
    {key: 'Escape',             action: 'blur',         desc: 'Unfocus current input'},
    {key: '?',                  action: 'showHelp',     desc: 'Show keyboard shortcuts'},
  ],
  actions: {},
  register(name, fn) { this.actions[name] = fn; },
  match(e) {
    return this.bindings.find(b =>
      b.key === e.key &&
      !!b.ctrl  === (e.ctrlKey && !e.metaKey) &&
      !!b.meta  === (e.metaKey && !e.ctrlKey) &&
      !!b.shift === e.shiftKey &&
      !!b.alt   === e.altKey
    );
  },
  formatKey(b) {
    const parts = [];
    if (b.ctrl || b.meta) parts.push(navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl');
    if (b.shift) parts.push('Shift');
    if (b.alt) parts.push('Alt');
    parts.push(b.key === ' ' ? 'Space' : b.key);
    return parts.join('+');
  }
};

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

function seedAdjust(delta) {
  const el = document.getElementById('seed');
  const v = parseInt(el.value, 10);
  if (isNaN(v) || v < 0) return;
  el.value = Math.max(0, v + delta);
  generate();
}

async function generate() {
  const btn = document.getElementById('genBtn');
  const box = document.getElementById('resultBox');
  const img = document.getElementById('resultImg');
  const ph = document.getElementById('placeholder');

  const fullPrompt = buildFullPrompt();
  const seed = document.getElementById('seed').value;
  const guidance = document.getElementById('guidance').value;
  const negPrompt = (document.getElementById('negPrompt').value || '').trim();

  btn.loading = true;
  box.classList.add('loading');
  if (ph) ph.style.display = 'none';

  const params = new URLSearchParams({
    prompt: fullPrompt, negative_prompt: negPrompt, seed, num_inference_steps: '50', guidance_scale: guidance
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

    document.getElementById('mPrompt').textContent = resp.headers.get('X-Prompt') || '';
    document.getElementById('mNeg').textContent = resp.headers.get('X-Negative-Prompt') || '(none)';
    document.getElementById('mModel').textContent = resp.headers.get('X-Model-Label') || resp.headers.get('X-Model') || 'sd-turbo';
    document.getElementById('mCfgType').textContent = resp.headers.get('X-Cfg-Type') || 'none';
    document.getElementById('mGuidance').textContent = respGuidance;
    document.getElementById('mSteps').textContent = resp.headers.get('X-Steps') || '50';
    document.getElementById('mSeed').textContent = actualSeed;
    document.getElementById('mTime').textContent = parseFloat(genTime).toFixed(3) + 's';
    document.getElementById('mIpAdapter').textContent = resp.headers.get('X-IP-Adapter') || 'off';
    const modCount = document.querySelectorAll('.chip[data-active="true"]').length;
    document.getElementById('mMods').textContent = modCount ? modCount + ' active' : 'none';
    document.getElementById('metaBox').style.display = '';

    const shareParams = new URLSearchParams({
      prompt: fullPrompt, negative_prompt: negPrompt, seed: actualSeed, guidance_scale: respGuidance
    });
    const shareUrl = window.location.origin + '/?' + shareParams.toString();
    document.getElementById('shareUrl').value = shareUrl;
    document.getElementById('shareBox').style.display = '';

    if (seed === '-1') document.getElementById('seed').value = actualSeed;
  } catch (e) {
    if (!modelLoading) showError(e.message);
  } finally {
    btn.loading = false;
    box.classList.remove('loading');
  }
}

let lastErrorMsg = '';
let lastErrorTime = 0;
function showError(msg) {
  const now = Date.now();
  if (msg === lastErrorMsg && now - lastErrorTime < 3000) return;
  lastErrorMsg = msg;
  lastErrorTime = now;
  const dlg = Object.assign(document.createElement('sl-alert'), {
    variant: 'danger', closable: true, duration: 5000,
    innerHTML: '<sl-icon slot="icon" name="exclamation-triangle"></sl-icon>' + msg
  });
  document.body.append(dlg);
  dlg.toast();
}

function copyShareUrl() {
  const el = document.getElementById('shareUrl');
  if (!el || !el.value) return;
  navigator.clipboard.writeText(el.value).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy URL', 1500);
  });
}

function showHelp() {
  let dlg = document.getElementById('shortcutHelp');
  if (dlg) { dlg.show(); return; }
  dlg = document.createElement('sl-dialog');
  dlg.id = 'shortcutHelp';
  dlg.label = 'Keyboard Shortcuts';
  const rows = shortcuts.bindings
    .filter(b => !(b.meta && b.action === 'generate'))
    .map(b => `<tr><td><kbd>${shortcuts.formatKey(b)}</kbd></td><td>${b.desc}</td></tr>`)
    .join('');
  dlg.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:var(--sl-font-size-small)">
    <tbody>${rows}</tbody></table>
    <style>#shortcutHelp kbd{background:var(--sl-color-neutral-100);padding:2px 8px;border-radius:4px;font-family:var(--sl-font-mono);font-size:var(--sl-font-size-x-small)}
    #shortcutHelp td{padding:6px 8px;border-bottom:1px solid var(--sl-color-neutral-100)}</style>`;
  document.body.append(dlg);
  dlg.show();
}

shortcuts.register('generate', generate);
shortcuts.register('seedUp', () => seedAdjust(1));
shortcuts.register('seedDown', () => seedAdjust(-1));
shortcuts.register('regenerate', generate);
shortcuts.register('copyUrl', copyShareUrl);
shortcuts.register('focusPrompt', () => document.getElementById('prompt').focus());
shortcuts.register('focusSeed', () => document.getElementById('seed').focus());
shortcuts.register('blur', () => document.activeElement?.blur());
shortcuts.register('showHelp', showHelp);

document.addEventListener('keydown', e => {
  if (document.getElementById('shortcutHelp')?.open && e.key === 'Escape') return;
  const inInput = e.target.closest('sl-textarea, sl-input, input, textarea');
  const b = shortcuts.match(e);
  if (!b) return;
  if (inInput && !b.ctrl && !b.meta && b.key !== 'Escape') return;
  e.preventDefault();
  const fn = shortcuts.actions[b.action];
  if (fn) fn();
});

document.getElementById('genBtn').addEventListener('click', generate);
document.getElementById('seedMinus').addEventListener('click', () => seedAdjust(-1));
document.getElementById('seedPlus').addEventListener('click', () => seedAdjust(1));
document.getElementById('seedRegen').addEventListener('click', generate);
document.getElementById('copyBtn').addEventListener('click', copyShareUrl);

let modelLoading = false;
document.getElementById('modelSelect').addEventListener('sl-change', async (e) => {
  const model = e.target.value;
  const badge = document.getElementById('modelStatus');
  const progress = document.getElementById('modelProgress');
  const select = e.target;
  badge.variant = 'warning';
  badge.textContent = 'loading...';
  progress.style.display = '';
  select.disabled = true;
  modelLoading = true;
  try {
    const resp = await fetch('/load_model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model})
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Failed to load model');
    badge.variant = 'success';
    badge.textContent = 'ready';
    const mModel = document.getElementById('mModel');
    if (mModel) mModel.textContent = model;
    if (data.guidance != null) document.getElementById('guidance').value = data.guidance;
    updateCnStatus();
  } catch (err) {
    badge.variant = 'danger';
    badge.textContent = 'error';
    showError(err.message);
  } finally {
    select.disabled = false;
    progress.style.display = 'none';
    modelLoading = false;
  }
});

// ControlNet handlers
const cnImageInput = document.getElementById('cnImageInput');
const cnUploadBtn = document.getElementById('cnUploadBtn');
const cnClearBtn = document.getElementById('cnClearBtn');
const cnStatus = document.getElementById('cnStatus');

cnImageInput.addEventListener('change', () => {
  cnUploadBtn.disabled = !cnImageInput.files.length;
});

cnUploadBtn.addEventListener('click', async () => {
  if (!cnImageInput.files.length) return;
  cnUploadBtn.loading = true;
  cnStatus.textContent = 'Uploading...';
  try {
    const formData = new FormData();
    formData.append('file', cnImageInput.files[0]);
    const resp = await fetch('/controlnet/upload', {method: 'POST', body: formData});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Upload failed');
    cnStatus.textContent = 'Control image active';
    document.getElementById('cnBadge').variant = 'success';
    document.getElementById('cnBadge').textContent = 'active';
  } catch (err) {
    cnStatus.textContent = '';
    showError(err.message);
  } finally {
    cnUploadBtn.loading = false;
  }
});

cnClearBtn.addEventListener('click', async () => {
  cnClearBtn.loading = true;
  try {
    const resp = await fetch('/controlnet/clear', {method: 'POST'});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Clear failed');
    cnStatus.textContent = 'Cleared';
    cnImageInput.value = '';
    cnUploadBtn.disabled = true;
    document.getElementById('cnBadge').variant = 'neutral';
    document.getElementById('cnBadge').textContent = 'available';
    setTimeout(() => { cnStatus.textContent = ''; }, 2000);
  } catch (err) {
    showError(err.message);
  } finally {
    cnClearBtn.loading = false;
  }
});

async function updateCnStatus() {
  try {
    const resp = await fetch('/models');
    if (!resp.ok) return;
    const data = await resp.json();
    const badge = document.getElementById('cnBadge');
    if (data.active_controlnet) {
      badge.variant = data.has_control_image ? 'success' : 'neutral';
      badge.textContent = data.has_control_image ? 'active' : 'available';
    } else {
      badge.variant = 'neutral';
      badge.textContent = 'available';
    }
  } catch (e) { /* ignore */ }
}

// IP-Adapter handlers
const ipImageInput = document.getElementById('ipImageInput');
const ipUploadBtn = document.getElementById('ipUploadBtn');
const ipClearBtn = document.getElementById('ipClearBtn');
const ipStatus = document.getElementById('ipStatus');

ipImageInput.addEventListener('change', () => {
  ipUploadBtn.disabled = !ipImageInput.files.length;
});

ipUploadBtn.addEventListener('click', async () => {
  if (!ipImageInput.files.length) return;
  ipUploadBtn.loading = true;
  ipStatus.textContent = 'Encoding reference...';
  try {
    const formData = new FormData();
    formData.append('file', ipImageInput.files[0]);
    const scale = document.getElementById('ipScale').value || '0.6';
    const resp = await fetch('/ip-adapter/upload?scale=' + scale, {method: 'POST', body: formData});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Upload failed');
    ipStatus.textContent = 'Reference active';
    document.getElementById('ipBadge').variant = 'success';
    document.getElementById('ipBadge').textContent = 'active';
  } catch (err) {
    ipStatus.textContent = '';
    showError(err.message);
  } finally {
    ipUploadBtn.loading = false;
  }
});

ipClearBtn.addEventListener('click', async () => {
  ipClearBtn.loading = true;
  try {
    const resp = await fetch('/ip-adapter/clear', {method: 'POST'});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Clear failed');
    ipStatus.textContent = 'Cleared';
    ipImageInput.value = '';
    ipUploadBtn.disabled = true;
    document.getElementById('ipBadge').variant = 'neutral';
    document.getElementById('ipBadge').textContent = 'available';
    setTimeout(() => { ipStatus.textContent = ''; }, 2000);
  } catch (err) {
    showError(err.message);
  } finally {
    ipClearBtn.loading = false;
  }
});

function renderSource(source) {
  const list = document.getElementById('sourceList');
  const item = document.createElement('div');
  item.className = 'source-item';
  item.dataset.sourceId = source.id;
  const label = document.createElement('span');
  label.className = 'source-label';
  label.title = source.url;
  label.innerHTML = `<strong>${source.label}</strong>`;
  label.addEventListener('click', () => {
    document.getElementById('prompt').value = source.url;
    generate();
  });
  const del = document.createElement('sl-icon-button');
  del.name = 'trash';
  del.label = 'Delete';
  del.addEventListener('click', async () => {
    const resp = await fetch('/sources/' + source.id, {method: 'DELETE'});
    if (resp.ok) item.remove();
  });
  item.append(label, del);
  list.appendChild(item);
}

async function loadSources() {
  try {
    const resp = await fetch('/sources');
    if (!resp.ok) return;
    const data = await resp.json();
    data.forEach(s => renderSource(s));
  } catch(e) { /* ignore */ }
}

document.getElementById('addSourceBtn').addEventListener('click', async () => {
  const input = document.getElementById('newSourceUrl');
  const url = input.value.trim();
  if (!url) return;
  try {
    const resp = await fetch('/sources', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url})
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Failed'); }
    const source = await resp.json();
    renderSource(source);
    input.value = '';
  } catch(e) { showError(e.message); }
});

document.getElementById('savePromptBtn').addEventListener('click', async () => {
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  try {
    const resp = await fetch('/sources/from-prompt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt})
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Failed'); }
    const source = await resp.json();
    renderSource(source);
  } catch(e) { showError(e.message); }
});

window.addEventListener('DOMContentLoaded', () => {
  loadSources();
  const p = new URLSearchParams(window.location.search);
  if (p.has('prompt')) {
    document.getElementById('prompt').value = p.get('prompt');
    if (p.has('negative_prompt')) document.getElementById('negPrompt').value = p.get('negative_prompt');
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
