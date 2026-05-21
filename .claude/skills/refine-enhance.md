---
name: refine-enhance
description: Iteratively refine the enhancement pipeline by generating images, evaluating them, and tuning the system prompt until output matches intent
user_invocable: true
---

# Refine Enhancement Pipeline

You are tuning the StreamDiffusion enhancement pipeline. The user gives you a prompt (and optionally a reference image path or detailed scene description). Your job is to cycle through enhance → generate → evaluate → adjust until the generated image closely matches the user's intent.

All configuration is data-driven — system prompts are stored in `enhance_prompts.json` and managed via API. No source code edits needed.

## API endpoints for enhance config

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET /enhance/config?scene=image` | Read the current system prompt for a scene |
| `GET /enhance/configs` | List all scenes with prompt previews |
| `PUT /enhance/config` | Update a scene's system prompt (persists to `enhance_prompts.json`) |
| `POST /enhance/config/reset` | Reset one or all scenes to defaults |

## Workflow

### 1. Parse the user's input
Extract:
- **prompt**: the raw generation prompt
- **reference**: path to a reference image (if provided) — read it to understand the target
- **description**: detailed text description of what the output should look like (if no reference)
- **scene**: which scene type (default "image")

### 2. Ensure a CFG-capable model is active
```bash
curl -s http://localhost:8384/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['active'], d['loading'])"
```
If not `dreamshaper-8`, switch to it:
```bash
curl -s -X POST http://localhost:8384/load_model -H 'Content-Type: application/json' -d '{"model":"dreamshaper-8"}'
```
Wait for loading to complete before generating.

### 3. Read the current enhance config
```bash
curl -s http://localhost:8384/enhance/config?scene=image | python3 -m json.tool
```
Note the current system prompt so you can make targeted adjustments.

### 4. Run the enhance → generate loop

Each iteration:

**a) Enhance the prompt:**
```bash
curl -s -X POST http://localhost:8384/enhance \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "...", "scene": "image"}' | python3 -m json.tool
```
Record the enhanced_prompt, negative_prompt, and notes.

**b) Generate an image with the enhanced prompt:**
```bash
curl -s -o /tmp/refine_iteration_N.png \
  'http://localhost:8384/generate?prompt=ENHANCED_PROMPT&negative_prompt=NEG_PROMPT&guidance_scale=1.2&seed=42'
```
Use a fixed seed for comparison across iterations. URL-encode the prompts.

**c) View the generated image:**
Use the Read tool to view `/tmp/refine_iteration_N.png`. Compare it against:
- The reference image (if provided)
- The user's description
- The original prompt's intent

**d) Evaluate and decide:**
Score the output on:
- **Subject accuracy**: Does it show what was described?
- **Composition**: Framing, layout, spatial relationships
- **Style/mood**: Lighting, color palette, atmosphere
- **Detail quality**: Sharpness, coherence, absence of artifacts
- **Prompt fidelity**: Did the enhancement preserve or lose important elements?

Report your evaluation to the user in 2-3 sentences.

**e) If the output is not satisfactory, adjust the system prompt via API:**

First read the current prompt:
```bash
curl -s http://localhost:8384/enhance/config?scene=image
```

Then update it with your fix:
```bash
curl -s -X PUT http://localhost:8384/enhance/config \
  -H 'Content-Type: application/json' \
  -d '{"scene": "image", "system_prompt": "...updated prompt..."}'
```

Common issues and fixes:
- **Enhancement drops key elements**: Add a rule like "Preserve ALL named subjects and their described attributes"
- **Too many quality boosters dilute the subject**: Reduce the quality booster rule, or cap them
- **Wrong emphasis weights**: Adjust the weight guidance (e.g., "Use (element:1.2-1.5) only for the primary subject")
- **Negative prompt too aggressive**: Refine negative prompt guidance
- **CLIP 77-token truncation**: The prompt is too long — add a rule to prioritize subject tokens and cut filler
- **Style/mood mismatch**: Add style-specific rules or examples

Changes take effect immediately on the next `/enhance` call — no server restart needed.

### 5. Iterate
Repeat steps 4a-4e, incrementing the iteration number. Typically 3-5 iterations are enough. After each adjustment, explain what you changed and why.

### 6. Verify with different seeds
Once satisfied, generate 2-3 images with different seeds to confirm the improvement is consistent:
```bash
for seed in 42 1337 9999; do
  curl -s -o /tmp/refine_verify_${seed}.png "http://localhost:8384/generate?prompt=...&guidance_scale=1.2&seed=${seed}"
done
```
View each and confirm quality.

### 7. Report
Summarize:
- What the original enhancement produced
- What changes you made to the system prompt (via API, persisted to `enhance_prompts.json`)
- What the final enhancement produces
- Show the final image(s)

To revert all changes: `curl -s -X POST http://localhost:8384/enhance/config/reset`

## Important notes
- Always use the same seed within a comparison iteration so changes are attributable to prompt differences, not randomness
- The CLIP tokenizer truncates at 77 tokens — if the enhanced prompt is long, front-loading matters enormously
- DreamShaper 8 with guidance 1.2 and 3 steps is the quality benchmark model
- If the user provides a reference image, read it first and describe what you see before starting the loop
- Keep edits surgical — don't rewrite the whole prompt each time, change one rule and test
- All changes persist to `enhance_prompts.json` — use `/enhance/config/reset` to restore defaults
