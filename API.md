# StreamDiffusion API Reference

Base URL: `http://<host>:8384`

## Models

### GET /models
Returns active model, loading state, available models, LoRAs, ControlNets, and IP-Adapter status.

```json
{
  "active": "sd-turbo",
  "loading": false,
  "active_controlnet": null,
  "has_control_image": false,
  "has_ip_adapter": true,
  "has_ip_reference": false,
  "models": {
    "sd-turbo": {"label": "SD-turbo", "desc": "...", "guidance": 1.0},
    "dreamshaper-8": {"label": "DreamShaper 8", "desc": "...", "guidance": 1.2},
    "lcm-dreamshaper": {"label": "LCM DreamShaper", "desc": "...", "guidance": 1.0}
  },
  "loras": { ... },
  "controlnets": { ... }
}
```

### POST /load_model
Switch the active model. Takes 5-30s depending on model.

**Request:**
```json
{"model": "dreamshaper-8", "controlnet": "canny"}
```
- `model` (required): key from `/models`
- `controlnet` (optional): key from controlnets list

**Response:**
```json
{"status": "loaded", "model": "dreamshaper-8", "controlnet": null, "guidance": 1.2}
```

Returns `409` if another model is already loading.

---

## Generation

### GET /generate
Generate an image from a text prompt.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | `"a photo of a cat"` | Text prompt (1-1000 chars) |
| `negative_prompt` | string | `""` | What to avoid (CFG models only) |
| `seed` | int | `-1` | Random seed (-1 = random) |
| `num_inference_steps` | int | `50` | Denoising steps (1-50) |
| `guidance_scale` | float | `1.0` | CFG scale (0-20, >1.0 enables negative prompt) |

**Response:** PNG image with metadata headers:

| Header | Example | Description |
|--------|---------|-------------|
| `X-Generation-Time` | `0.0389` | Seconds |
| `X-Seed` | `42` | Actual seed used |
| `X-Steps` | `50` | Steps used |
| `X-Guidance-Scale` | `1.2` | CFG scale |
| `X-Model` | `dreamshaper-8` | Model key |
| `X-Model-Label` | `DreamShaper 8` | Display name |
| `X-Cfg-Type` | `full` | `none` or `full` |
| `X-Prompt` | `a cat` | Prompt (truncated 200 chars) |
| `X-Negative-Prompt` | `blurry` | Negative prompt or `(none)` |
| `X-IP-Adapter` | `active` | `active` or `off` |
| `X-Width` | `512` | Output width |
| `X-Height` | `512` | Output height |

Returns `400` for invalid params, `503` during model loading.

**Example:**
```bash
curl -o image.png 'http://localhost:8384/generate?prompt=a+red+car&seed=42&guidance_scale=1.2'
```

---

## ControlNet

Controls spatial structure by extracting edges from a reference image.

### POST /controlnet/upload
Upload a reference image for Canny edge ControlNet.

**Request:** `multipart/form-data` with `file` field (image).

**Response:**
```json
{"status": "ok", "message": "ControlNet image set (canny edges)"}
```

### POST /controlnet/clear
Remove the control image and disable ControlNet guidance.

**Response:**
```json
{"status": "ok", "message": "ControlNet image cleared"}
```

---

## IP-Adapter

Transfers identity/style from a reference image into generations. Only works with SD 1.5 models (DreamShaper, LCM DreamShaper). SD-turbo is incompatible.

### POST /ip-adapter/upload
Upload a reference image (face, character, style) for IP-Adapter.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `scale` | float | `0.6` | Influence strength (0-1) |

**Request:** `multipart/form-data` with `file` field (image).

**Response:**
```json
{"status": "ok", "message": "IP-Adapter reference set (scale=0.6)"}
```

Returns `400` if IP-Adapter not loaded (incompatible model).

### POST /ip-adapter/clear
Remove the reference image.

**Response:**
```json
{"status": "ok", "message": "IP-Adapter reference cleared"}
```

---

## Sources

Text sources that can be loaded into the prompt.

### GET /sources
List all sources.

**Response:** `[{"id": "uuid", "url": "...", "label": "..."}]`

### POST /sources
Add a new source.

**Request:**
```json
{"url": "https://example.com", "label": "My Source"}
```
- `label` defaults to `url` if omitted.

**Response:** `201` with `{"id": "uuid", "url": "...", "label": "..."}`

### DELETE /sources/{source_id}
Delete a source by ID.

**Response:** `{"status": "deleted", "id": "..."}`

Returns `404` if not found.

### POST /sources/from-prompt
Save the current prompt text as a source.

**Request:**
```json
{"prompt": "a beautiful sunset"}
```

**Response:** `201` with `{"id": "uuid", "url": "a beautiful sunset", "label": "Saved prompt"}`

---

## Model Compatibility Matrix

| Feature | sd-turbo | dreamshaper-8 | lcm-dreamshaper |
|---------|----------|---------------|-----------------|
| Negative Prompt | No (cfg=none) | Yes (cfg=full) | Yes (cfg=full) |
| ControlNet | No | Yes | Yes |
| IP-Adapter | No (dim mismatch) | Yes | Yes |
| Speed | ~40ms | ~400ms | ~200ms |
| Steps | 1 | 3 | 3 |
