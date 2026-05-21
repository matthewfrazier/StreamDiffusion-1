# StreamDiffusion API Reference

Base URL: `http://<host>:8384`

## Quickstart

```bash
# Install dependencies (requires CUDA-capable GPU)
pip install -e .

# Set your Anthropic API key for prompt enhancement and title generation
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# Start the server
python app.py
```

The server starts on `http://127.0.0.1:8384`. For remote access, use Tailscale or an HTTPS reverse proxy.

## Web UI

### GET /
Serves the full interactive web application — a single-page app with real-time image generation, model switching, preset chips, prompt enhancement, sources, ControlNet, and IP-Adapter controls. No build step; the HTML is served inline from the backend.

Open `http://<host>:8384/` in a browser to use.

### GET /embed/compose
Serves a lightweight, self-contained compose widget designed for embedding in other applications via `<iframe>`. Includes scene selector, preset chips, active pills, `#` autocomplete, and enhance button. See [Embeddable Compose Component](#embeddable-compose-component) for the full postMessage protocol.

```html
<iframe src="http://<host>:8384/embed/compose?scene=music"
        style="width:100%;height:400px;border:none"></iframe>
```

---

## Scenes & Presets

The API supports multiple creative scenes (image, music, sound), each with its own preset library and prompt enhancement behavior.

### GET /presets
Returns the preset library for a scene.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `scene` | string | `"image"` | Scene type: `image`, `music`, `sound` |

**Response:**
```json
{
  "label": "Music Generation",
  "categories": [
    {
      "name": "Genre",
      "presets": [
        {"label": "Ambient", "value": "ambient, atmospheric, ethereal pads, drone"},
        {"label": "Electronic", "value": "electronic, synthesizer, sequenced, digital"}
      ]
    },
    {"name": "Mood", "presets": [...]},
    {"name": "Instrumentation", "presets": [...]},
    {"name": "Tempo", "presets": [...]}
  ]
}
```

Available scenes:
- `image` — Style, Composition, Lighting presets for SD 1.5
- `music` — Genre, Mood, Instrumentation, Tempo presets for music generation
- `sound` — Type, Character, Environment, Dynamics presets for sound design

### Scene CRUD

Full management of scene type definitions. Custom scenes integrate with `/presets`, `/enhance`, and the embed widget.

**Scene key format:** lowercase alphanumeric with hyphens, 2-50 chars (`^[a-z0-9][a-z0-9-]{0,48}[a-z0-9]$`).

**Scene schema:**
```json
{
  "label": "My Custom Scene",
  "categories": [
    {
      "name": "Category Name",
      "presets": [
        {"label": "Preset Label", "value": "comma-separated modifier tokens"}
      ]
    }
  ]
}
```

#### GET /scenes
List all scenes with summary stats.

**Response:**
```json
{
  "image": {"label": "Image Generation", "category_count": 3, "preset_count": 27},
  "music": {"label": "Music Generation", "category_count": 4, "preset_count": 33},
  "sound": {"label": "Sound Design", "category_count": 4, "preset_count": 28}
}
```

#### GET /scenes/{key}
Get full scene definition.

**Response:**
```json
{
  "key": "music",
  "label": "Music Generation",
  "categories": [
    {"name": "Genre", "presets": [{"label": "Ambient", "value": "ambient, atmospheric, ethereal pads, drone"}, ...]},
    {"name": "Mood", "presets": [...]},
    ...
  ]
}
```

Returns `404` if not found.

#### PUT /scenes/{key}
Create or replace a scene. Preset labels must be unique across all categories.

**Request:** Scene schema (see above).

**Response:** `200` (replaced) or `201` (created) with full scene definition.

Returns `400` for invalid key or duplicate preset labels.

#### PATCH /scenes/{key}
Partial update. Supports these operations (any combination in one request):

| Field | Type | Effect |
|-------|------|--------|
| `label` | string | Update display name |
| `add_category` | `{name, presets}` | Add a new category |
| `remove_category` | string | Remove category by name |
| `add_presets` | `{category, presets}` | Append presets to a category (skips duplicates) |
| `remove_presets` | `{category, labels}` | Remove presets by label from a category |

**Example — add presets to an existing category:**
```json
{
  "add_presets": {
    "category": "Genre",
    "presets": [
      {"label": "Jazz", "value": "jazz, swing, improvisation, brass"}
    ]
  }
}
```

**Response:** Full updated scene definition.

Returns `404` if scene not found, `409` if adding a category that already exists.

#### DELETE /scenes/{key}
Delete a custom scene. Built-in scenes (image, music, sound) can also be deleted but will not persist across restarts.

**Response:**
```json
{"status": "deleted", "key": "my-scene"}
```

Returns `404` if not found.

#### POST /scenes/generate
Generate a scene from a flat list of labels. Send descriptive terms — moods, genres, textures, instruments, tempos, environments — and the LLM figures out what categories make sense and maps each label in.

**Request:**
```json
{
  "labels": ["jazzy piano", "vinyl crackle", "rainy day", "mellow drums", "coffee shop", "late night"],
  "key": "lofi-study",
  "save": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `labels` | string[] | yes | Flat list of descriptive terms (at least 1) |
| `key` | string | no | Override the LLM-suggested scene key |
| `save` | bool | no | If `true`, immediately register the scene (available via `/presets`, `/enhance`, embed widget) |

**Response:** `201`
```json
{
  "key": "lofi-study",
  "saved": true,
  "label": "Lo-Fi Study Beats",
  "categories": [
    {
      "name": "Vibe",
      "presets": [
        {"label": "Rainy Day", "value": "rain ambience, gentle patter, cozy, introspective"},
        {"label": "Late Night", "value": "nocturnal, dim, quiet intensity, solitary focus"},
        {"label": "Coffee Shop", "value": "warm, murmur, espresso machine, ambient chatter"}
      ]
    },
    {
      "name": "Instrumentation",
      "presets": [
        {"label": "Jazzy Piano", "value": "rhodes piano, jazz chords, seventh chords, mellow keys"},
        {"label": "Vinyl Crackle", "value": "vinyl crackle, tape hiss, analog warmth, lo-fi noise"},
        {"label": "Mellow Drums", "value": "soft kick, brushed snare, lo-fi percussion, lazy groove"}
      ]
    }
  ]
}
```

Every input label is placed into exactly one category. The LLM infers 3-5 categories by semantic fit and expands each label into generation tokens. When `save: false` (default), the response is a preview the client can tweak before `PUT /scenes/{key}`.

Returns `422` for empty labels, `500` if `ANTHROPIC_API_KEY` is not set.

### Enhance Configuration

The system prompts that drive `/enhance` are data-driven — stored in `enhance_prompts.json` and fully manageable via API. No source code changes needed to tune enhancement behavior.

#### GET /enhance/configs
List all configured scenes with prompt previews (first 100 chars).

**Response:**
```json
{
  "image": "You are a Stable Diffusion 1.5 prompt engineer. Rewrite natural language image descriptions into...",
  "music": "You are a music generation prompt engineer. Rewrite natural language music descriptions into opt...",
  "sound": "You are a sound design prompt engineer for AI audio generation. Rewrite natural language sound d..."
}
```

#### GET /enhance/config
Get the full system prompt for a scene.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `scene` | string | `"image"` | Scene to retrieve |

**Response:**
```json
{
  "scene": "image",
  "system_prompt": "You are a Stable Diffusion 1.5 prompt engineer..."
}
```

Returns `404` if scene has no enhance config.

#### PUT /enhance/config
Update the system prompt for a scene. Persists to `enhance_prompts.json` and takes effect immediately.

**Request:**
```json
{
  "scene": "image",
  "system_prompt": "You are a Stable Diffusion 1.5 prompt engineer...\n\nRules:\n- ..."
}
```

**Response:**
```json
{
  "scene": "image",
  "system_prompt": "...",
  "saved": true
}
```

Returns `400` for missing or too-short system_prompt.

#### POST /enhance/config/reset
Reset system prompt(s) to built-in defaults.

**Request:**
```json
{"scene": "image"}
```
Omit `scene` to reset all scenes.

**Response:**
```json
{"status": "reset", "scenes": ["image"]}
```

### POST /enhance
Rewrite a prompt using LLM enhancement, optimized per scene. Returns suggested preset chips.

**Request:**
```json
{
  "prompt": "a dark ambient track for a horror scene",
  "scene": "music",
  "context": {
    "mood": "terrifying",
    "environment": "abandoned hospital"
  }
}
```
- `prompt` (required): raw prompt text
- `scene` (optional, default `"image"`): selects the enhancement strategy and modifier library
- `context` (optional): structured key-value pairs folded into the prompt

**Response:**
```json
{
  "enhanced_prompt": "Dark ambient horror soundtrack, 60-80 BPM, minor key...",
  "negative_prompt": "uplifting, major key, bright, distortion, clipping...",
  "notes": "Added specific instrumentation and sonic descriptors...",
  "original_prompt": "a dark ambient track for a horror scene",
  "suggested_chips": ["Ambient", "Tense", "Synth Pads", "Slow / Adagio"]
}
```

`suggested_chips` contains 0-4 preset labels from the scene's library. Clients should activate matching presets in the UI.

Returns `400` for empty prompt, `500` if `ANTHROPIC_API_KEY` is not set.

---

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

Saved prompts with LLM-generated titles and associated presets.

### GET /sources
List all sources.

**Response:**
```json
[{"id": "uuid", "url": "prompt text...", "label": "Crystalline Coyotes Dusk", "chips": ["Cinematic", "Dramatic"], "scene": "image"}]
```

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
Save a prompt with its active presets. Generates an LLM title automatically.

**Request:**
```json
{
  "prompt": "three coyotes with crystalline protrusions...",
  "chips": ["Cinematic", "Dramatic"],
  "scene": "image"
}
```

**Response:** `201`
```json
{
  "id": "uuid",
  "url": "three coyotes with crystalline protrusions...",
  "label": "Crystalline Coyotes Dusk",
  "chips": ["Cinematic", "Dramatic"],
  "scene": "image"
}
```

---

## Embeddable Compose Component

### GET /embed/compose
Returns a self-contained HTML5 compose widget that can be embedded via `<iframe>`. Includes the scene selector, preset chips, active pills, `#` autocomplete, and enhance button. Communicates with the host page via `postMessage`. No external dependencies — all CSS and JS are inlined.

```html
<iframe src="http://<host>:8384/embed/compose?scene=music"
        style="width:100%;height:400px;border:none"></iframe>
```

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `scene` | string | `"image"` | Initial scene |

**postMessage events (iframe → parent):**
| Event | Payload | Trigger |
|-------|---------|---------|
| `prompt` | `{type: "prompt", prompt, chips, scene}` | User clicks Generate |
| `enhanced` | `{type: "enhanced", enhanced_prompt, negative_prompt, notes, suggested_chips}` | Enhance completes |

**postMessage events (parent → iframe):**
| Event | Payload | Effect |
|-------|---------|--------|
| `setScene` | `{type: "setScene", scene: "sound"}` | Switch active scene, reload presets |
| `setPrompt` | `{type: "setPrompt", prompt: "...", chips: [...]}` | Load prompt text and activate matching chips |

**Integration example:**
```javascript
const iframe = document.querySelector('iframe');

// Listen for compose events
window.addEventListener('message', (e) => {
  if (e.data.type === 'prompt') {
    // User composed a prompt — send to your generation backend
    console.log(e.data.prompt, e.data.chips, e.data.scene);
  }
});

// Load a saved prompt into the compose widget
iframe.contentWindow.postMessage({
  type: 'setPrompt',
  prompt: 'ambient horror soundscape',
  chips: ['Ambient', 'Tense']
}, '*');
```

---

## Model Compatibility Matrix

| Feature | sd-turbo | dreamshaper-8 | lcm-dreamshaper |
|---------|----------|---------------|-----------------|
| Negative Prompt | No (cfg=none) | Yes (cfg=full) | Yes (cfg=full) |
| ControlNet | No | Yes | Yes |
| IP-Adapter | No (dim mismatch) | Yes | Yes |
| Speed | ~40ms | ~400ms | ~200ms |
| Steps | 1 | 3 | 3 |
