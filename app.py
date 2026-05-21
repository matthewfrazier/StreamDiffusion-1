import asyncio
import functools
import gc
import io
import json
import os
import random
import sys
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

import torch
from fastapi import FastAPI, File, HTTPException, Path, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, validator
from typing import List, Optional
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

SCENE_PRESETS = {
    "image": {
        "label": "Image Generation",
        "categories": [
            {"name": "Style", "presets": [
                {"label": "Photorealistic", "value": "photorealistic, 8K, detailed"},
                {"label": "Oil Painting", "value": "oil painting, impressionist, brushstrokes"},
                {"label": "Watercolor", "value": "watercolor painting, soft edges, wet media"},
                {"label": "Digital Art", "value": "digital art, concept art, trending on artstation"},
                {"label": "Pencil Sketch", "value": "pencil sketch, graphite, hand-drawn"},
                {"label": "Anime", "value": "anime style, cel shaded, vibrant colors"},
                {"label": "Pixel Art", "value": "pixel art, 16-bit, retro game"},
                {"label": "3D Render", "value": "3D render, octane render, volumetric lighting"},
                {"label": "Cinematic", "value": "cinematic, anamorphic lens, film grain, color graded"},
                {"label": "Minimalist", "value": "minimalist, clean lines, flat design"},
            ]},
            {"name": "Composition", "presets": [
                {"label": "Close-up", "value": "close-up, macro, tight framing"},
                {"label": "Wide Angle", "value": "wide angle, expansive, establishing shot"},
                {"label": "Bird's Eye", "value": "bird's eye view, top-down, aerial"},
                {"label": "Low Angle", "value": "low angle, worm's eye view, looking up"},
                {"label": "Portrait", "value": "portrait, centered subject, shallow depth of field"},
                {"label": "Panoramic", "value": "panoramic, ultrawide, landscape orientation"},
                {"label": "Symmetrical", "value": "symmetrical, centered composition, balanced"},
                {"label": "Rule of Thirds", "value": "rule of thirds, off-center subject"},
                {"label": "Bokeh", "value": "bokeh background, blurred background, subject isolation"},
                {"label": "Tilt-Shift", "value": "tilt-shift, miniature effect, selective focus"},
            ]},
            {"name": "Lighting", "presets": [
                {"label": "Golden Hour", "value": "golden hour, warm sunlight, long shadows"},
                {"label": "Dramatic", "value": "dramatic lighting, chiaroscuro, high contrast"},
                {"label": "Soft/Diffused", "value": "soft diffused light, overcast, even lighting"},
                {"label": "Neon", "value": "neon lights, cyberpunk, glowing"},
                {"label": "Moonlit", "value": "moonlight, nighttime, cool blue tones"},
                {"label": "Studio", "value": "studio lighting, rim light, professional"},
                {"label": "Backlit", "value": "backlit, silhouette, halo light"},
            ]},
        ],
    },
    "music": {
        "label": "Music Generation",
        "categories": [
            {"name": "Genre", "presets": [
                {"label": "Ambient", "value": "ambient, atmospheric, ethereal pads, drone"},
                {"label": "Electronic", "value": "electronic, synthesizer, sequenced, digital"},
                {"label": "Classical", "value": "classical, orchestral, composed, acoustic ensemble"},
                {"label": "Jazz", "value": "jazz, swing, improvisation, blue notes"},
                {"label": "Hip-Hop", "value": "hip-hop, boom bap, sampled, rhythmic"},
                {"label": "Rock", "value": "rock, electric guitar, drums, distortion"},
                {"label": "Folk", "value": "folk, acoustic, organic, traditional instruments"},
                {"label": "Cinematic Score", "value": "cinematic score, film music, orchestral, sweeping"},
                {"label": "Lo-Fi", "value": "lo-fi, vinyl crackle, mellow, warm tape saturation"},
                {"label": "Synthwave", "value": "synthwave, retro 80s, analog synth, arpeggiated"},
            ]},
            {"name": "Mood", "presets": [
                {"label": "Uplifting", "value": "uplifting, bright, major key, hopeful"},
                {"label": "Melancholic", "value": "melancholic, sad, minor key, wistful"},
                {"label": "Tense", "value": "tense, suspenseful, dissonant, building anxiety"},
                {"label": "Peaceful", "value": "peaceful, calm, serene, gentle"},
                {"label": "Energetic", "value": "energetic, driving, high tempo, powerful"},
                {"label": "Mysterious", "value": "mysterious, dark, enigmatic, unsettling"},
                {"label": "Epic", "value": "epic, grandiose, triumphant, soaring"},
                {"label": "Intimate", "value": "intimate, quiet, close, personal"},
            ]},
            {"name": "Instrumentation", "presets": [
                {"label": "Piano", "value": "piano, keys, grand piano, ivory"},
                {"label": "Strings", "value": "strings, violin, cello, orchestral strings"},
                {"label": "Synth Pads", "value": "synth pads, warm pads, evolving texture"},
                {"label": "Acoustic Guitar", "value": "acoustic guitar, fingerpicked, nylon string"},
                {"label": "Full Orchestra", "value": "full orchestra, tutti, brass and strings"},
                {"label": "Drums/Percussion", "value": "drums, percussion, rhythmic, beat"},
                {"label": "Bass", "value": "bass, sub bass, deep, low-end"},
                {"label": "Vocals", "value": "vocals, choir, harmonies, voice"},
                {"label": "Brass", "value": "brass, trumpet, horn, trombone"},
            ]},
            {"name": "Tempo", "presets": [
                {"label": "Slow / Adagio", "value": "slow tempo, adagio, 60-80 BPM"},
                {"label": "Medium / Andante", "value": "medium tempo, andante, 90-110 BPM"},
                {"label": "Fast / Allegro", "value": "fast tempo, allegro, 120-150 BPM"},
                {"label": "Building", "value": "building tempo, accelerando, crescendo"},
                {"label": "Sparse", "value": "sparse arrangement, minimal, space between notes"},
                {"label": "Dense/Layered", "value": "dense arrangement, layered, complex texture"},
            ]},
        ],
    },
    "sound": {
        "label": "Sound Design",
        "categories": [
            {"name": "Type", "presets": [
                {"label": "Foley", "value": "foley, natural recording, realistic, physical"},
                {"label": "Ambient", "value": "ambient soundscape, environmental, background"},
                {"label": "SFX", "value": "sound effect, designed, processed, layered"},
                {"label": "UI/UX", "value": "UI sound, notification, interface, click, confirmation"},
                {"label": "Nature", "value": "nature sounds, field recording, organic, outdoors"},
                {"label": "Industrial", "value": "industrial, machinery, mechanical, factory"},
                {"label": "Sci-Fi", "value": "sci-fi, futuristic, electronic, alien, laser"},
                {"label": "Impact", "value": "impact, hit, slam, collision, explosion"},
            ]},
            {"name": "Character", "presets": [
                {"label": "Organic", "value": "organic, natural, acoustic, unprocessed"},
                {"label": "Synthetic", "value": "synthetic, digital, generated, electronic"},
                {"label": "Metallic", "value": "metallic, resonant, ringing, steel"},
                {"label": "Warm", "value": "warm, rounded, soft, analog"},
                {"label": "Sharp", "value": "sharp, crisp, transient, bright attack"},
                {"label": "Deep", "value": "deep, rumble, sub-frequency, low"},
                {"label": "Textured", "value": "textured, grainy, layered, complex surface"},
            ]},
            {"name": "Environment", "presets": [
                {"label": "Indoor", "value": "indoor, room tone, reflections, enclosed"},
                {"label": "Outdoor", "value": "outdoor, open air, wind, spacious"},
                {"label": "Underwater", "value": "underwater, submerged, muffled, bubbles"},
                {"label": "Cave/Tunnel", "value": "cave, tunnel, long reverb, echo"},
                {"label": "Forest", "value": "forest, woodland, leaves, wildlife"},
                {"label": "Urban", "value": "urban, city, traffic, crowd, concrete"},
                {"label": "Abstract", "value": "abstract, non-spatial, surreal, impossible"},
            ]},
            {"name": "Dynamics", "presets": [
                {"label": "Sustained", "value": "sustained, long, continuous, held"},
                {"label": "Transient", "value": "transient, short, percussive, impulse"},
                {"label": "Rising", "value": "rising, ascending, building, crescendo"},
                {"label": "Falling", "value": "falling, descending, decaying, diminuendo"},
                {"label": "Rhythmic", "value": "rhythmic, pulsing, repeating, loopable"},
                {"label": "One-Shot", "value": "one-shot, single event, non-repeating"},
            ]},
        ],
    },
}

ENHANCE_PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhance_prompts.json")

def _load_enhance_bases():
    try:
        with open(ENHANCE_PROMPTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_enhance_bases(bases):
    with open(ENHANCE_PROMPTS_FILE, "w") as f:
        json.dump(bases, f, indent=2)

SCENE_ENHANCE_BASES = _load_enhance_bases()


def build_enhance_prompt(scene):
    presets = SCENE_PRESETS.get(scene, SCENE_PRESETS["image"])
    labels = [p["label"] for cat in presets["categories"] for p in cat["presets"]]
    modifier_list = ", ".join(labels)
    base = SCENE_ENHANCE_BASES.get(scene, SCENE_ENHANCE_BASES["image"])
    return base + f'- Also return a "suggested_modifiers" array of 0-4 names from this exact list: {modifier_list}. Only suggest modifiers that genuinely improve the prompt.\n- Return ONLY valid JSON with keys: "prompt", "negative_prompt", "notes", "suggested_modifiers"\n- Do NOT wrap in markdown code fences. Return raw JSON only.\n'


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


class PresetItem(BaseModel):
    label: str = Field(..., min_length=1, max_length=50)
    value: str = Field(..., min_length=1, max_length=500)


class PresetCategory(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    presets: List[PresetItem] = Field(..., min_items=1)


class SceneDefinition(BaseModel):
    label: str = Field(..., min_length=1, max_length=100)
    categories: List[PresetCategory] = Field(..., min_items=1)

    @validator("categories")
    def unique_category_names(cls, v):
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError("Category names must be unique")
        return v


@app.get("/presets")
async def get_presets(scene: str = Query("image")):
    if scene not in SCENE_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown scene: {scene}. Available: {', '.join(SCENE_PRESETS.keys())}")
    return JSONResponse(SCENE_PRESETS[scene])


SCENE_KEY_RE = __import__("re").compile(r"^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$")


@app.get("/scenes")
async def list_scenes():
    return JSONResponse({
        k: {"label": v["label"], "category_count": len(v["categories"]),
            "preset_count": sum(len(c["presets"]) for c in v["categories"])}
        for k, v in SCENE_PRESETS.items()
    })


@app.get("/scenes/{scene_key}")
async def get_scene(scene_key: str = Path(...)):
    if scene_key not in SCENE_PRESETS:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene_key}")
    return JSONResponse({"key": scene_key, **SCENE_PRESETS[scene_key]})


@app.put("/scenes/{scene_key}")
async def put_scene(scene_key: str = Path(...), body: SceneDefinition = ...):
    if not SCENE_KEY_RE.match(scene_key):
        raise HTTPException(status_code=400, detail="Scene key must be lowercase alphanumeric with hyphens, 2-50 chars")
    all_labels = []
    for cat in body.categories:
        for p in cat.presets:
            if p.label in all_labels:
                raise HTTPException(status_code=400, detail=f"Duplicate preset label: {p.label}")
            all_labels.append(p.label)
    SCENE_PRESETS[scene_key] = {
        "label": body.label,
        "categories": [
            {"name": c.name, "presets": [{"label": p.label, "value": p.value} for p in c.presets]}
            for c in body.categories
        ],
    }
    status = 200 if scene_key in SCENE_PRESETS else 201
    return JSONResponse({"key": scene_key, **SCENE_PRESETS[scene_key]}, status_code=status)


@app.patch("/scenes/{scene_key}")
async def patch_scene(scene_key: str = Path(...), body: dict = ...):
    if scene_key not in SCENE_PRESETS:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene_key}")
    scene = SCENE_PRESETS[scene_key]
    if "label" in body and isinstance(body["label"], str):
        scene["label"] = body["label"]
    if "add_category" in body and isinstance(body["add_category"], dict):
        cat = body["add_category"]
        if "name" in cat and "presets" in cat:
            existing_names = [c["name"] for c in scene["categories"]]
            if cat["name"] in existing_names:
                raise HTTPException(status_code=409, detail=f"Category already exists: {cat['name']}")
            scene["categories"].append({"name": cat["name"], "presets": cat["presets"]})
    if "remove_category" in body and isinstance(body["remove_category"], str):
        scene["categories"] = [c for c in scene["categories"] if c["name"] != body["remove_category"]]
    if "add_presets" in body and isinstance(body["add_presets"], dict):
        cat_name = body["add_presets"].get("category")
        presets = body["add_presets"].get("presets", [])
        for cat in scene["categories"]:
            if cat["name"] == cat_name:
                existing = {p["label"] for p in cat["presets"]}
                for p in presets:
                    if p.get("label") and p.get("value") and p["label"] not in existing:
                        cat["presets"].append({"label": p["label"], "value": p["value"]})
                break
    if "remove_presets" in body and isinstance(body["remove_presets"], dict):
        cat_name = body["remove_presets"].get("category")
        labels = set(body["remove_presets"].get("labels", []))
        for cat in scene["categories"]:
            if cat["name"] == cat_name:
                cat["presets"] = [p for p in cat["presets"] if p["label"] not in labels]
                break
    return JSONResponse({"key": scene_key, **scene})


@app.delete("/scenes/{scene_key}")
async def delete_scene(scene_key: str = Path(...)):
    if scene_key not in SCENE_PRESETS:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene_key}")
    del SCENE_PRESETS[scene_key]
    return JSONResponse({"status": "deleted", "key": scene_key})


@app.get("/enhance/config")
async def get_enhance_config(scene: str = Query("image")):
    if scene not in SCENE_ENHANCE_BASES:
        raise HTTPException(status_code=404, detail=f"No enhance config for scene: {scene}")
    return JSONResponse({"scene": scene, "system_prompt": SCENE_ENHANCE_BASES[scene]})


@app.get("/enhance/configs")
async def list_enhance_configs():
    return JSONResponse({k: v[:100] + "..." for k, v in SCENE_ENHANCE_BASES.items()})


@app.put("/enhance/config")
async def put_enhance_config(body: dict):
    scene = body.get("scene")
    system_prompt = body.get("system_prompt")
    if not scene or not isinstance(scene, str):
        raise HTTPException(status_code=400, detail="scene is required")
    if not system_prompt or not isinstance(system_prompt, str) or len(system_prompt.strip()) < 10:
        raise HTTPException(status_code=400, detail="system_prompt is required (min 10 chars)")
    SCENE_ENHANCE_BASES[scene] = system_prompt.strip()
    _save_enhance_bases(SCENE_ENHANCE_BASES)
    return JSONResponse({"scene": scene, "system_prompt": SCENE_ENHANCE_BASES[scene], "saved": True})


@app.post("/enhance/config/reset")
async def reset_enhance_config(body: dict = None):
    scene = (body or {}).get("scene")
    defaults = {
        "image": "You are a Stable Diffusion 1.5 prompt engineer. Rewrite natural language image descriptions into optimized SD 1.5 prompts.\n\nRules:\n- Front-load the subject, then details (pose, expression, clothing), then environment, then style/quality tags.\n- Use emphasis weights like (element:1.3) for important visual elements. Use sparingly (2-4 weighted terms max).\n- Add quality boosters: masterpiece, best quality, highly detailed, sharp focus — but only if appropriate for the medium.\n- Keep the total prompt under 200 tokens.\n- Generate a matching negative prompt for common SD 1.5 failures (bad anatomy, blurry, watermark, etc.).\n- Incorporate ALL provided context fields naturally into the prompt.\n- The \"notes\" field should be 1-2 sentences explaining what you changed and why.",
        "music": "You are a music generation prompt engineer. Rewrite natural language music descriptions into optimized prompts for AI music generation models (MusicGen, Stable Audio, Udio).\n\nRules:\n- Front-load genre/style, then tempo and key, then instrumentation, then mood and dynamics.\n- Use precise musical terminology: BPM ranges, time signatures, key signatures, dynamics markings.\n- Add quality descriptors: professional mix, mastered, high fidelity, studio recording — where appropriate.\n- Keep the total prompt under 200 tokens.\n- Generate a matching negative prompt (distortion, clipping, off-key, low quality, noise, etc.).\n- Incorporate ALL provided context fields naturally into the prompt.\n- The \"notes\" field should be 1-2 sentences explaining what you changed and why.",
        "sound": "You are a sound design prompt engineer for AI audio generation. Rewrite natural language sound descriptions into optimized prompts.\n\nRules:\n- Front-load the sound type and source, then environment/space, then texture and character, then temporal dynamics.\n- Use precise audio terminology: frequency range, attack/decay/sustain/release, reverb characteristics, stereo width.\n- Add quality descriptors: high fidelity, clean recording, studio quality, 48kHz — where appropriate.\n- Keep the total prompt under 200 tokens.\n- Generate a matching negative prompt (noise, distortion, clipping, artifacts, low quality, etc.).\n- Incorporate ALL provided context fields naturally into the prompt.\n- The \"notes\" field should be 1-2 sentences explaining what you changed and why.",
    }
    if scene:
        if scene not in defaults:
            raise HTTPException(status_code=404, detail=f"No default for scene: {scene}")
        SCENE_ENHANCE_BASES[scene] = defaults[scene]
    else:
        SCENE_ENHANCE_BASES.update(defaults)
    _save_enhance_bases(SCENE_ENHANCE_BASES)
    scenes_reset = [scene] if scene else list(defaults.keys())
    return JSONResponse({"status": "reset", "scenes": scenes_reset})


class GenerateSceneRequest(BaseModel):
    labels: List[str] = Field(..., min_items=1)
    key: Optional[str] = None
    save: bool = False


GENERATE_SCENE_SYSTEM = """\
You are an expert at organizing creative concepts for AI generation tools.
You receive a flat list of labels — short descriptive terms like moods, genres, textures, instruments, styles, tempos, environments, etc.
Your job is to figure out what categories make sense and place each label into the best-fit category.

Rules:
- Infer 3-5 categories from the labels (e.g., Mood, Genre, Texture, Tempo, Instrumentation, Environment, Style).
- Place every input label into exactly one category — do not drop any.
- A label can only appear once. If it could fit multiple categories, pick the strongest semantic match.
- For each label, generate a "value" field: comma-separated generation tokens (3-8 tokens) that expand the label into useful generation detail.
- If a category ends up with only 1 label, merge it into the closest related category.
- Also produce a "label" for the scene itself (2-5 words, title case).
- Also produce a "key" — a lowercase kebab-case slug (2-50 chars, letters/digits/hyphens only).
- Return ONLY valid JSON with keys: "key", "label", "categories"
- categories is an array of {"name": "...", "presets": [{"label": "...", "value": "..."}]}
- Do NOT wrap in markdown code fences. Return raw JSON only.
"""


@app.post("/scenes/generate")
async def generate_scene(req: GenerateSceneRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY environment variable is not set")

    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package is not installed. Run: uv pip install anthropic")

    labels = [l.strip() for l in req.labels if l.strip()]
    if not labels:
        raise HTTPException(status_code=400, detail="At least one non-empty label is required")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=GENERATE_SCENE_SYSTEM,
            messages=[{"role": "user", "content": f"Organize these labels into a scene:\n\n{json.dumps(labels)}"}],
        )

        response_text = message.content[0].text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise HTTPException(status_code=500, detail="Failed to parse scene generation response as JSON")

        scene_key = req.key or result.get("key", "custom-scene")
        if not SCENE_KEY_RE.match(scene_key):
            scene_key = "custom-scene"
        scene_label = result.get("label", "Custom Scene")
        categories = result.get("categories", [])

        if not categories:
            raise HTTPException(status_code=500, detail="LLM returned no categories")

        scene_def = {"label": scene_label, "categories": categories}

        if req.save:
            SCENE_PRESETS[scene_key] = scene_def

        return JSONResponse({
            "key": scene_key,
            "saved": req.save,
            **scene_def,
        }, status_code=201)

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Scene generation failed: {str(e)}")


class LoadModelRequest(BaseModel):
    model: str
    controlnet: Optional[str] = None


class AddSourceRequest(BaseModel):
    url: str
    label: Optional[str] = None


class SavePromptRequest(BaseModel):
    prompt: str
    chips: Optional[list] = None
    scene: Optional[str] = "image"


class EnhanceRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    scene: Optional[str] = "image"


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
            "X-Prompt": prompt[:200].replace("\n", " ").replace("\r", ""),
            "X-Negative-Prompt": neg[:200].replace("\n", " ").replace("\r", "") if neg else "(none)",
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
    label = "Saved prompt"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                system="Return ONLY a 2-4 word title summarizing this image prompt. No quotes, no punctuation, no explanation.",
                messages=[{"role": "user", "content": req.prompt[:300]}],
            )
            label = msg.content[0].text.strip().strip('"').strip("'")[:40]
        except Exception:
            label = req.prompt[:30].split(",")[0].strip()
    scene = req.scene if req.scene in SCENE_PRESETS else "image"
    source = {
        "id": str(uuid.uuid4()),
        "url": req.prompt,
        "label": label,
        "chips": req.chips or [],
        "scene": scene,
    }
    sources.append(source)
    return JSONResponse(source, status_code=201)


@app.post("/enhance")
async def enhance_prompt(req: EnhanceRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")
    scene = req.scene if req.scene in SCENE_PRESETS else "image"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable is not set"
        )

    try:
        import anthropic
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="anthropic package is not installed. Run: uv pip install anthropic"
        )

    # Build the user message with context
    user_parts = [f"Rewrite this into an optimized SD 1.5 prompt:\n\n\"{req.prompt.strip()}\""]
    if req.context:
        ctx_lines = []
        for key, value in req.context.items():
            if value and str(value).strip():
                ctx_lines.append(f"- {key}: {value}")
        if ctx_lines:
            user_parts.append("\nContext:\n" + "\n".join(ctx_lines))

    user_message = "\n".join(user_parts)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=build_enhance_prompt(scene),
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract text content from the response
        response_text = message.content[0].text.strip()

        # Parse the JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it's wrapped in markdown
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse enhancement response as JSON"
                )

        enhanced_prompt = result.get("prompt", req.prompt)
        negative_prompt = result.get("negative_prompt", "")
        notes = result.get("notes", "")
        suggested_chips = result.get("suggested_modifiers", [])

        return JSONResponse({
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "notes": notes,
            "original_prompt": req.prompt,
            "suggested_chips": suggested_chips,
        })

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


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
@media(max-width:768px){.layout{grid-template-columns:minmax(0,1fr);gap:12px}body{padding:8px}.sidebar{order:2;min-width:0}.main{order:1;min-width:0}.result{min-height:unset}}
.sidebar{display:flex;flex-direction:column;gap:12px}
.main{display:flex;flex-direction:column;gap:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{touch-action:manipulation;-webkit-tap-highlight-color:transparent;border:1px solid var(--sl-color-neutral-300);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-700);padding:4px 14px;border-radius:9999px;font:inherit;font-size:var(--sl-font-size-small);cursor:pointer;user-select:none;transition:background .15s,color .15s,border-color .15s}
.chip[data-active="true"]{background:var(--sl-color-primary-600);color:#fff;border-color:var(--sl-color-primary-600)}
.chip:focus-visible{outline:2px solid var(--sl-color-primary-600);outline-offset:2px}
.active-pills{display:flex;flex-wrap:wrap;gap:6px;min-height:0;margin-bottom:6px}
.active-pills:empty{display:none;margin-bottom:0}
.active-pill{touch-action:manipulation;-webkit-tap-highlight-color:transparent;display:inline-flex;align-items:center;gap:6px;background:var(--sl-color-primary-600);color:#fff;padding:6px 12px;border-radius:9999px;font-size:var(--sl-font-size-small);cursor:default;min-height:32px}
.active-pill .pill-x{touch-action:manipulation;-webkit-tap-highlight-color:transparent;cursor:pointer;opacity:.7;font-size:18px;line-height:1;padding:4px;margin:-4px -4px -4px 0;min-width:28px;min-height:28px;display:flex;align-items:center;justify-content:center}
.active-pill .pill-x:hover{opacity:1}
.autocomplete-wrap{position:relative}
.autocomplete-list{position:absolute;left:0;right:0;bottom:100%;max-height:200px;overflow-y:auto;background:var(--sl-color-neutral-0);border:1px solid var(--sl-color-neutral-300);border-radius:var(--sl-border-radius-medium);box-shadow:var(--sl-shadow-large);z-index:100;display:none}
@media(max-width:768px){.autocomplete-list{bottom:auto;top:100%}}
.autocomplete-list.open{display:block}
.autocomplete-item{touch-action:manipulation;-webkit-tap-highlight-color:transparent;padding:12px 14px;cursor:pointer;font-size:var(--sl-font-size-small);min-height:44px;display:flex;align-items:center}
.autocomplete-item:hover,.autocomplete-item.highlighted{background:var(--sl-color-primary-100);color:var(--sl-color-primary-700)}
.autocomplete-item .ac-cat{color:var(--sl-color-neutral-400);font-size:var(--sl-font-size-x-small);margin-left:6px}
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
      <div class="active-pills" id="activePills"></div>
      <div class="autocomplete-wrap">
        <div class="autocomplete-list" id="acList"></div>
        <sl-textarea id="prompt" rows="2" value="a beautiful landscape, mountains, sunset" resize="auto" placeholder="Describe your image... type # for presets"></sl-textarea>
        <div style="display:flex;gap:6px;margin-top:6px">
          <sl-button variant="primary" size="small" id="genBtn" style="flex:1">Generate</sl-button>
          <sl-button size="small" variant="success" id="enhanceBtn" style="flex:1">Enhance</sl-button>
          <sl-button size="small" variant="text" id="savePromptBtn">Save</sl-button>
        </div>
      </div>
      <div id="enhanceNotes" class="muted" style="display:none;margin-top:4px;padding:2px 4px;font-style:italic"></div>
      <sl-details summary="Context (for prompt enhancement)" id="contextDetails" style="margin-top:8px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
          <sl-input id="ctxMood" size="small" placeholder="Mood" label="Mood"></sl-input>
          <sl-input id="ctxGenre" size="small" placeholder="Genre" label="Genre"></sl-input>
          <sl-input id="ctxLocation" size="small" placeholder="Location" label="Location"></sl-input>
          <sl-input id="ctxTimeOfDay" size="small" placeholder="Time of day" label="Time of Day"></sl-input>
          <sl-input id="ctxEnvironment" size="small" placeholder="Environment" label="Environment"></sl-input>
          <sl-input id="ctxCamera" size="small" placeholder="Camera angle" label="Camera"></sl-input>
          <sl-input id="ctxMedium" size="small" placeholder="Medium / style" label="Medium"></sl-input>
        </div>
      </sl-details>
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

    <div id="chipSections"></div>

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

let PRESETS = [];
let currentScene = 'image';

async function loadPresets(scene) {
  currentScene = scene;
  try {
    const resp = await fetch('/presets?scene=' + scene);
    if (!resp.ok) return;
    const data = await resp.json();
    PRESETS = [];
    const container = document.getElementById('chipSections');
    container.innerHTML = '';
    const outer = document.createElement('sl-details');
    outer.summary = data.categories.map(c => c.name).join(', ');
    outer.open = true;
    data.categories.forEach((cat, ci) => {
      const details = document.createElement('sl-details');
      details.summary = cat.name;
      if (ci === 0) details.open = true;
      const chips = document.createElement('div');
      chips.className = 'chips';
      cat.presets.forEach(p => {
        PRESETS.push({label: p.label, value: p.value, cat: cat.name});
        const btn = document.createElement('button');
        btn.className = 'chip';
        btn.dataset.v = p.value;
        btn.dataset.active = 'false';
        btn.textContent = p.label;
        btn.addEventListener('click', () => {
          btn.dataset.active = btn.dataset.active === 'true' ? 'false' : 'true';
          syncPills();
        });
        chips.appendChild(btn);
      });
      details.appendChild(chips);
      outer.appendChild(details);
    });
    container.appendChild(outer);
    syncPills();
  } catch(e) { console.error('Failed to load presets', e); }
}

let acItems = [];
let acIdx = -1;

function getPromptTextarea() {
  return document.getElementById('prompt').shadowRoot?.querySelector('textarea');
}

function syncPills() {
  const container = document.getElementById('activePills');
  container.innerHTML = '';
  document.querySelectorAll('.chip[data-active="true"]').forEach(chip => {
    const pill = document.createElement('span');
    pill.className = 'active-pill';
    const label = chip.textContent.trim();
    pill.textContent = label;
    const x = document.createElement('span');
    x.className = 'pill-x';
    x.innerHTML = '&times;';
    x.addEventListener('click', () => {
      chip.dataset.active = 'false';
      syncPills();
    });
    pill.appendChild(x);
    container.appendChild(pill);
  });
}

function getHashQuery() {
  const ta = getPromptTextarea();
  if (!ta) return null;
  const pos = ta.selectionStart;
  const text = ta.value.substring(0, pos);
  const hashIdx = text.lastIndexOf('#');
  if (hashIdx === -1) return null;
  if (hashIdx > 0 && !/[\\s,]/.test(text[hashIdx - 1])) return null;
  const query = text.substring(hashIdx + 1);
  if (query.includes('\\n')) return null;
  return { query: query.toLowerCase(), start: hashIdx, end: pos };
}

function updateAutocomplete() {
  const hq = getHashQuery();
  const list = document.getElementById('acList');
  if (!hq) { list.classList.remove('open'); acItems = []; acIdx = -1; return; }
  const activeLabels = new Set(
    [...document.querySelectorAll('.chip[data-active="true"]')].map(c => c.textContent.trim())
  );
  const matches = PRESETS.filter(p =>
    !activeLabels.has(p.label) && p.label.toLowerCase().includes(hq.query)
  ).slice(0, 8);
  if (!matches.length) { list.classList.remove('open'); acItems = []; acIdx = -1; return; }
  list.innerHTML = matches.map((m, i) =>
    '<div class="autocomplete-item' + (i === 0 ? ' highlighted' : '') + '" data-idx="' + i + '">' +
    m.label + '<span class="ac-cat">' + m.cat + '</span></div>'
  ).join('');
  acItems = matches;
  acIdx = 0;
  list.classList.add('open');
}

function highlightAcItem() {
  document.querySelectorAll('.autocomplete-item').forEach((el, i) => {
    el.classList.toggle('highlighted', i === acIdx);
  });
}

function selectAutocomplete(preset) {
  const hq = getHashQuery();
  const promptEl = document.getElementById('prompt');
  const ta = getPromptTextarea();
  if (hq && ta) {
    const before = ta.value.substring(0, hq.start);
    const after = ta.value.substring(hq.end);
    promptEl.value = (before + after).replace(/,\\s*$/, '').replace(/^\\s*,\\s*/, '');
  }
  const chip = [...document.querySelectorAll('.chip')].find(c => c.textContent.trim() === preset.label);
  if (chip) chip.dataset.active = 'true';
  syncPills();
  document.getElementById('acList').classList.remove('open');
  acItems = [];
  acIdx = -1;
  promptEl.focus();
}

function activateChipByLabel(label) {
  const chip = [...document.querySelectorAll('.chip')].find(c => c.textContent.trim() === label);
  if (chip && chip.dataset.active !== 'true') {
    chip.dataset.active = 'true';
    return true;
  }
  return false;
}

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
    if (activeSourceId) shareParams.set('source', activeSourceId);
    const shareUrl = window.location.origin + '/?' + shareParams.toString();
    document.getElementById('shareUrl').value = shareUrl;
    document.getElementById('shareBox').style.display = '';
    history.replaceState(null, '', '/?' + shareParams.toString());

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

document.getElementById('genBtn').addEventListener('click', () => { activeSourceId = null; generate(); });
document.getElementById('prompt').addEventListener('sl-input', () => { activeSourceId = null; updateAutocomplete(); });
document.getElementById('seedMinus').addEventListener('click', () => seedAdjust(-1));
document.getElementById('seedPlus').addEventListener('click', () => seedAdjust(1));
document.getElementById('seedRegen').addEventListener('click', generate);
document.getElementById('copyBtn').addEventListener('click', copyShareUrl);

document.getElementById('prompt').addEventListener('keydown', (e) => {
  const list = document.getElementById('acList');
  if (!list.classList.contains('open')) return;
  if (e.key === 'ArrowDown') { e.preventDefault(); acIdx = Math.min(acIdx + 1, acItems.length - 1); highlightAcItem(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); acIdx = Math.max(acIdx - 1, 0); highlightAcItem(); }
  else if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); if (acIdx >= 0 && acIdx < acItems.length) selectAutocomplete(acItems[acIdx]); }
  else if (e.key === 'Escape') { list.classList.remove('open'); acItems = []; acIdx = -1; }
});

document.getElementById('acList').addEventListener('click', (e) => {
  const item = e.target.closest('.autocomplete-item');
  if (!item) return;
  const idx = parseInt(item.dataset.idx, 10);
  if (idx >= 0 && idx < acItems.length) selectAutocomplete(acItems[idx]);
});

let modelLoading = false;
let activeSourceId = null;
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
  const preview = source.url.length > 60 ? source.url.substring(0, 60) + '...' : source.url;
  label.innerHTML = `<strong>${source.label}</strong><br><span class="muted" style="font-size:var(--sl-font-size-x-small);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block">${preview}</span>`;
  label.addEventListener('click', async () => {
    document.getElementById('prompt').value = source.url;
    document.querySelectorAll('.chip').forEach(c => c.dataset.active = 'false');
    if (source.chips && source.chips.length) {
      source.chips.forEach(chipLabel => activateChipByLabel(chipLabel));
    }
    syncPills();
    activeSourceId = source.id;
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
  const btn = document.getElementById('savePromptBtn');
  btn.loading = true;
  try {
    const chips = [...document.querySelectorAll('.chip[data-active="true"]')].map(c => c.textContent.trim());
    const resp = await fetch('/sources/from-prompt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, chips, scene: currentScene})
    });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.detail || 'Failed'); }
    const source = await resp.json();
    renderSource(source);
  } catch(e) { showError(e.message); }
  finally { btn.loading = false; }
});

function getContext() {
  const fields = {
    mood: document.getElementById('ctxMood'),
    genre: document.getElementById('ctxGenre'),
    location: document.getElementById('ctxLocation'),
    time_of_day: document.getElementById('ctxTimeOfDay'),
    environment: document.getElementById('ctxEnvironment'),
    camera: document.getElementById('ctxCamera'),
    medium: document.getElementById('ctxMedium'),
  };
  const ctx = {};
  let hasAny = false;
  for (const [k, el] of Object.entries(fields)) {
    const v = (el.value || '').trim();
    if (v) { ctx[k] = v; hasAny = true; }
  }
  return hasAny ? ctx : null;
}

document.getElementById('enhanceBtn').addEventListener('click', async () => {
  const promptEl = document.getElementById('prompt');
  const prompt = promptEl.value.trim();
  if (!prompt) { showError('Enter a prompt to enhance'); return; }
  const btn = document.getElementById('enhanceBtn');
  const notesEl = document.getElementById('enhanceNotes');
  btn.loading = true;
  notesEl.style.display = 'none';
  try {
    const body = { prompt, scene: currentScene };
    const ctx = getContext();
    if (ctx) body.context = ctx;
    const resp = await fetch('/enhance', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      throw new Error(err?.detail || 'Enhancement failed: ' + resp.status);
    }
    const data = await resp.json();
    promptEl.value = data.enhanced_prompt;
    if (data.negative_prompt) {
      document.getElementById('negPrompt').value = data.negative_prompt;
    }
    if (data.suggested_chips && data.suggested_chips.length) {
      data.suggested_chips.forEach(label => activateChipByLabel(label));
      syncPills();
    }
    if (data.notes) {
      notesEl.textContent = data.notes;
      notesEl.style.display = '';
    }
  } catch(e) {
    showError(e.message);
  } finally {
    btn.loading = false;
  }
});

window.addEventListener('DOMContentLoaded', async () => {
  await loadPresets('image');
  await loadSources();
  const p = new URLSearchParams(window.location.search);
  if (p.has('source')) {
    activeSourceId = p.get('source');
    const resp = await fetch('/sources');
    if (resp.ok) {
      const sources = await resp.json();
      const match = sources.find(s => s.id === activeSourceId);
      if (match) document.getElementById('prompt').value = match.url;
    }
  }
  if (p.has('prompt')) {
    document.getElementById('prompt').value = p.get('prompt');
  }
  if (p.has('negative_prompt')) document.getElementById('negPrompt').value = p.get('negative_prompt');
  if (p.has('seed')) document.getElementById('seed').value = p.get('seed');
  if (p.has('guidance_scale')) document.getElementById('guidance').value = p.get('guidance_scale');
  if (p.has('prompt') || p.has('source')) generate();
});
</script>
</body>
</html>"""


@app.get("/embed/compose", response_class=HTMLResponse)
async def embed_compose(scene: str = Query("image")):
    if scene not in SCENE_PRESETS:
        scene = "image"
    return f"""<!DOCTYPE html>
<html lang="en" class="sl-theme-dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/themes/dark.css">
<script type="module" src="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/shoelace-autoloader.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:var(--sl-font-sans);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-900);padding:12px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px}}
.chip{{touch-action:manipulation;-webkit-tap-highlight-color:transparent;border:1px solid var(--sl-color-neutral-300);background:var(--sl-color-neutral-0);color:var(--sl-color-neutral-700);padding:4px 14px;border-radius:9999px;font:inherit;font-size:var(--sl-font-size-small);cursor:pointer;user-select:none;transition:background .15s,color .15s,border-color .15s}}
.chip[data-active="true"]{{background:var(--sl-color-primary-600);color:#fff;border-color:var(--sl-color-primary-600)}}
.active-pills{{display:flex;flex-wrap:wrap;gap:6px;min-height:0;margin-bottom:6px}}
.active-pills:empty{{display:none;margin-bottom:0}}
.active-pill{{touch-action:manipulation;display:inline-flex;align-items:center;gap:6px;background:var(--sl-color-primary-600);color:#fff;padding:6px 12px;border-radius:9999px;font-size:var(--sl-font-size-small);min-height:32px}}
.active-pill .pill-x{{cursor:pointer;opacity:.7;font-size:18px;padding:4px;min-width:28px;min-height:28px;display:flex;align-items:center;justify-content:center}}
.active-pill .pill-x:hover{{opacity:1}}
.autocomplete-wrap{{position:relative}}
.autocomplete-list{{position:absolute;left:0;right:0;top:100%;max-height:200px;overflow-y:auto;background:var(--sl-color-neutral-0);border:1px solid var(--sl-color-neutral-300);border-radius:var(--sl-border-radius-medium);box-shadow:var(--sl-shadow-large);z-index:100;display:none}}
.autocomplete-list.open{{display:block}}
.autocomplete-item{{touch-action:manipulation;padding:12px 14px;cursor:pointer;font-size:var(--sl-font-size-small);min-height:44px;display:flex;align-items:center}}
.autocomplete-item:hover,.autocomplete-item.highlighted{{background:var(--sl-color-primary-100);color:var(--sl-color-primary-700)}}
.autocomplete-item .ac-cat{{color:var(--sl-color-neutral-400);font-size:var(--sl-font-size-x-small);margin-left:6px}}
.muted{{font-size:var(--sl-font-size-small);color:var(--sl-color-neutral-500)}}
</style>
</head>
<body>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
  <sl-select id="sceneSelect" size="small" value="{scene}" style="width:130px" hoist>
  </sl-select>
  <span class="muted" id="sceneLabel"></span>
</div>
<div class="active-pills" id="activePills"></div>
<div class="autocomplete-wrap">
  <div class="autocomplete-list" id="acList"></div>
  <div style="display:flex;gap:8px;align-items:flex-end">
    <sl-textarea id="prompt" rows="3" resize="auto" placeholder="Describe... type # for presets" style="flex:1"></sl-textarea>
    <div style="display:flex;flex-direction:column;gap:4px">
      <sl-button variant="primary" id="genBtn">Generate</sl-button>
      <sl-button size="small" variant="success" id="enhanceBtn">Enhance</sl-button>
    </div>
  </div>
</div>
<div id="chipSections" style="margin-top:12px"></div>
<script type="module">
let PRESETS = [], currentScene = '{scene}', acItems = [], acIdx = -1;
function getTA() {{ return document.getElementById('prompt').shadowRoot?.querySelector('textarea'); }}
function syncPills() {{
  const c = document.getElementById('activePills'); c.innerHTML = '';
  document.querySelectorAll('.chip[data-active="true"]').forEach(chip => {{
    const pill = document.createElement('span'); pill.className = 'active-pill';
    pill.textContent = chip.textContent.trim();
    const x = document.createElement('span'); x.className = 'pill-x'; x.innerHTML = '&times;';
    x.addEventListener('click', () => {{ chip.dataset.active = 'false'; syncPills(); }});
    pill.appendChild(x); c.appendChild(pill);
  }});
}}
function activateChipByLabel(label) {{
  const chip = [...document.querySelectorAll('.chip')].find(c => c.textContent.trim() === label);
  if (chip && chip.dataset.active !== 'true') {{ chip.dataset.active = 'true'; return true; }} return false;
}}
function getActiveChipLabels() {{ return [...document.querySelectorAll('.chip[data-active="true"]')].map(c => c.textContent.trim()); }}
function getActiveModifiers() {{ return [...document.querySelectorAll('.chip[data-active="true"]')].map(c => c.dataset.v); }}
async function loadPresets(scene) {{
  currentScene = scene;
  const resp = await fetch('/presets?scene=' + scene); if (!resp.ok) return;
  const data = await resp.json(); PRESETS = [];
  const container = document.getElementById('chipSections'); container.innerHTML = '';
  const outer = document.createElement('sl-details'); outer.summary = data.categories.map(c => c.name).join(', '); outer.open = true;
  data.categories.forEach((cat, ci) => {{
    const d = document.createElement('sl-details'); d.summary = cat.name; if (ci === 0) d.open = true;
    const chips = document.createElement('div'); chips.className = 'chips';
    cat.presets.forEach(p => {{
      PRESETS.push({{label: p.label, value: p.value, cat: cat.name}});
      const btn = document.createElement('button'); btn.className = 'chip'; btn.dataset.v = p.value; btn.dataset.active = 'false'; btn.textContent = p.label;
      btn.addEventListener('click', () => {{ btn.dataset.active = btn.dataset.active === 'true' ? 'false' : 'true'; syncPills(); }});
      chips.appendChild(btn);
    }}); d.appendChild(chips); outer.appendChild(d);
  }}); container.appendChild(outer);
  document.getElementById('sceneLabel').textContent = data.label; syncPills();
}}
function getHashQuery() {{
  const ta = getTA(); if (!ta) return null;
  const pos = ta.selectionStart, text = ta.value.substring(0, pos), hi = text.lastIndexOf('#');
  if (hi === -1) return null; if (hi > 0 && !/[\\s,]/.test(text[hi-1])) return null;
  const q = text.substring(hi+1); if (q.includes('\\n')) return null;
  return {{query: q.toLowerCase(), start: hi, end: pos}};
}}
function updateAutocomplete() {{
  const hq = getHashQuery(), list = document.getElementById('acList');
  if (!hq) {{ list.classList.remove('open'); acItems = []; acIdx = -1; return; }}
  const active = new Set(getActiveChipLabels());
  const matches = PRESETS.filter(p => !active.has(p.label) && p.label.toLowerCase().includes(hq.query)).slice(0,8);
  if (!matches.length) {{ list.classList.remove('open'); acItems = []; acIdx = -1; return; }}
  list.innerHTML = matches.map((m,i) => '<div class="autocomplete-item'+(i===0?' highlighted':'')+'" data-idx="'+i+'">'+m.label+'<span class="ac-cat">'+m.cat+'</span></div>').join('');
  acItems = matches; acIdx = 0; list.classList.add('open');
}}
function selectAutocomplete(preset) {{
  const hq = getHashQuery(), el = document.getElementById('prompt'), ta = getTA();
  if (hq && ta) {{ el.value = (ta.value.substring(0,hq.start)+ta.value.substring(hq.end)).replace(/,\\s*$/,'').replace(/^\\s*,\\s*/,''); }}
  const chip = [...document.querySelectorAll('.chip')].find(c => c.textContent.trim() === preset.label);
  if (chip) chip.dataset.active = 'true'; syncPills();
  document.getElementById('acList').classList.remove('open'); acItems = []; acIdx = -1; el.focus();
}}
document.getElementById('prompt').addEventListener('sl-input', () => updateAutocomplete());
document.getElementById('prompt').addEventListener('keydown', (e) => {{
  const list = document.getElementById('acList'); if (!list.classList.contains('open')) return;
  if (e.key==='ArrowDown') {{ e.preventDefault(); acIdx=Math.min(acIdx+1,acItems.length-1); document.querySelectorAll('.autocomplete-item').forEach((el,i)=>el.classList.toggle('highlighted',i===acIdx)); }}
  else if (e.key==='ArrowUp') {{ e.preventDefault(); acIdx=Math.max(acIdx-1,0); document.querySelectorAll('.autocomplete-item').forEach((el,i)=>el.classList.toggle('highlighted',i===acIdx)); }}
  else if (e.key==='Enter') {{ e.preventDefault(); e.stopPropagation(); if (acIdx>=0&&acIdx<acItems.length) selectAutocomplete(acItems[acIdx]); }}
  else if (e.key==='Escape') {{ list.classList.remove('open'); acItems=[]; acIdx=-1; }}
}});
document.getElementById('acList').addEventListener('click', (e) => {{
  const item = e.target.closest('.autocomplete-item'); if (!item) return;
  const idx = parseInt(item.dataset.idx,10); if (idx>=0&&idx<acItems.length) selectAutocomplete(acItems[idx]);
}});
document.getElementById('sceneSelect').addEventListener('sl-change', (e) => {{
  document.getElementById('activePills').innerHTML = ''; loadPresets(e.target.value);
}});
document.getElementById('genBtn').addEventListener('click', () => {{
  const prompt = document.getElementById('prompt').value.trim();
  const mods = getActiveModifiers();
  const full = mods.length ? prompt + ', ' + mods.join(', ') : prompt;
  window.parent.postMessage({{type:'prompt', prompt: full, chips: getActiveChipLabels(), scene: currentScene}}, '*');
}});
document.getElementById('enhanceBtn').addEventListener('click', async () => {{
  const prompt = document.getElementById('prompt').value.trim(); if (!prompt) return;
  const btn = document.getElementById('enhanceBtn'); btn.loading = true;
  try {{
    const resp = await fetch('/enhance', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{prompt, scene: currentScene}})}});
    if (!resp.ok) return; const data = await resp.json();
    document.getElementById('prompt').value = data.enhanced_prompt;
    if (data.suggested_chips) {{ data.suggested_chips.forEach(l => activateChipByLabel(l)); syncPills(); }}
    window.parent.postMessage({{type:'enhanced', ...data}}, '*');
  }} finally {{ btn.loading = false; }}
}});
window.addEventListener('message', (e) => {{
  if (e.data?.type === 'setScene') {{ document.getElementById('sceneSelect').value = e.data.scene; loadPresets(e.data.scene); }}
  if (e.data?.type === 'setPrompt') {{
    document.getElementById('prompt').value = e.data.prompt || '';
    document.querySelectorAll('.chip').forEach(c => c.dataset.active = 'false');
    if (e.data.chips) e.data.chips.forEach(l => activateChipByLabel(l));
    syncPills();
  }}
}});
async function loadScenes() {{
  try {{
    const resp = await fetch('/scenes'); if (!resp.ok) return;
    const scenes = await resp.json();
    const sel = document.getElementById('sceneSelect'); sel.innerHTML = '';
    for (const [key, info] of Object.entries(scenes)) {{
      const opt = document.createElement('sl-option'); opt.value = key; opt.textContent = info.label;
      sel.appendChild(opt);
    }}
    sel.value = currentScene;
  }} catch(e) {{}}
}}
(async () => {{ await loadScenes(); await loadPresets('{scene}'); }})();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8384)
