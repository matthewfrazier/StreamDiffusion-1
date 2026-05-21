"""Playwright integration tests for the StreamDiffusion web app (Shoelace UI).

Covers chip toggle behavior (including mobile touch), seed controls,
generation flow, input validation, and model rendering.

Run:  .venv/bin/python -m pytest test_ui.py -v
"""

import io
import pytest
import requests
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://127.0.0.1:8384"

MOBILE_DEVICE = {
    "viewport": {"width": 412, "height": 915},
    "user_agent": "Mozilla/5.0 (Linux; Android 14; Pixel 9) AppleWebKit/537.36",
    "has_touch": True,
    "is_mobile": True,
}


def wait_for_shoelace(page):
    """Wait for Shoelace components to be defined in the custom elements registry."""
    page.wait_for_function(
        "() => customElements.get('sl-button') !== undefined",
        timeout=15000,
    )


def get_chip_active(chip):
    return chip.get_attribute("data-active") == "true"


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def desktop_page(browser):
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(BASE_URL)
    page.wait_for_load_state("domcontentloaded")
    wait_for_shoelace(page)
    yield page
    ctx.close()


@pytest.fixture
def mobile_page(browser):
    ctx = browser.new_context(**MOBILE_DEVICE)
    page = ctx.new_page()
    page.goto(BASE_URL)
    page.wait_for_load_state("domcontentloaded")
    wait_for_shoelace(page)
    yield page
    ctx.close()


class TestChipToggleDesktop:
    def test_chip_starts_inactive(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        assert not get_chip_active(chip)

    def test_click_toggles_on(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        chip.click()
        assert get_chip_active(chip)

    def test_click_toggles_off(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        chip.click()
        assert get_chip_active(chip)
        chip.click()
        assert not get_chip_active(chip)

    def test_rapid_toggle_5x(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        for i in range(5):
            chip.click()
        assert get_chip_active(chip)  # odd number -> active

    def test_multiple_chips_independent(self, desktop_page):
        chips = desktop_page.locator(".chip")
        chips.nth(0).click()
        chips.nth(2).click()
        assert get_chip_active(chips.nth(0))
        assert not get_chip_active(chips.nth(1))
        assert get_chip_active(chips.nth(2))

    def test_deselect_one_leaves_other(self, desktop_page):
        chips = desktop_page.locator(".chip")
        chips.nth(0).click()
        chips.nth(1).click()
        assert get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1))
        chips.nth(0).click()
        assert not get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1))


class TestChipToggleMobile:
    def test_tap_toggles_on(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        chip.tap()
        assert get_chip_active(chip)

    def test_tap_toggles_off(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        chip.tap()
        assert get_chip_active(chip)
        chip.tap()
        assert not get_chip_active(chip)

    def test_tap_rapid_toggle_5x(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        for i in range(5):
            chip.tap()
        assert get_chip_active(chip)

    def test_tap_multiple_chips(self, mobile_page):
        chips = mobile_page.locator(".chip")
        chips.nth(0).tap()
        chips.nth(2).tap()
        assert get_chip_active(chips.nth(0))
        assert not get_chip_active(chips.nth(1))
        assert get_chip_active(chips.nth(2))

    def test_tap_deselect_one_leaves_other(self, mobile_page):
        chips = mobile_page.locator(".chip")
        chips.nth(0).tap()
        chips.nth(1).tap()
        chips.nth(0).tap()
        assert not get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1))

    def test_tap_toggle_10x_ends_off(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        for _ in range(10):
            chip.tap()
        assert not get_chip_active(chip)

    def test_tap_on_then_off_zero_active_in_section(self, mobile_page):
        """Tap a chip on, tap it off, confirm zero chips active in its section."""
        section_chips = mobile_page.locator(".chips .chip")
        chip = section_chips.first
        chip.tap()
        assert get_chip_active(chip)
        chip.tap()
        assert not get_chip_active(chip)
        active_count = section_chips.evaluate_all(
            "chips => chips.filter(c => c.dataset.active === 'true').length"
        )
        assert active_count == 0

    def test_tap_second_chip_does_not_deactivate_first(self, mobile_page):
        """Tapping chip B must not change chip A's state."""
        chips = mobile_page.locator(".chips .chip")
        chips.nth(0).tap()
        assert get_chip_active(chips.nth(0))
        chips.nth(1).tap()
        assert get_chip_active(chips.nth(0)), "first chip lost its active state"
        assert get_chip_active(chips.nth(1))

    def test_tap_sequence_activates_then_deactivates_each(self, mobile_page):
        """Activate 3 chips, deactivate each individually, confirm zero remain."""
        chips = mobile_page.locator(".chips .chip")
        chips.nth(0).tap()
        chips.nth(1).tap()
        chips.nth(2).tap()
        assert get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1))
        assert get_chip_active(chips.nth(2))
        chips.nth(0).tap()
        assert not get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1))
        assert get_chip_active(chips.nth(2))
        chips.nth(1).tap()
        assert not get_chip_active(chips.nth(1))
        assert get_chip_active(chips.nth(2))
        chips.nth(2).tap()
        assert not get_chip_active(chips.nth(2))
        active_count = chips.evaluate_all(
            "chips => chips.filter(c => c.dataset.active === 'true').length"
        )
        assert active_count == 0


class TestChipToggleRealTouch:
    """Simulate real touch event chains (touchstart→touchend→click) to catch
    double-fire bugs that Playwright's .tap() doesn't reproduce."""

    def _real_tap(self, page, locator):
        """Dispatch touchstart + touchend + click like a real finger."""
        box = locator.bounding_box()
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        locator.evaluate(
            """(el, coords) => {
                const t = new Touch({identifier: Date.now(), target: el,
                    clientX: coords.x, clientY: coords.y});
                el.dispatchEvent(new TouchEvent('touchstart',
                    {bubbles: true, cancelable: true, touches: [t], changedTouches: [t]}));
                el.dispatchEvent(new TouchEvent('touchend',
                    {bubbles: true, cancelable: true, touches: [], changedTouches: [t]}));
                el.dispatchEvent(new MouseEvent('click',
                    {bubbles: true, cancelable: true, clientX: coords.x, clientY: coords.y}));
            }""",
            {"x": cx, "y": cy},
        )

    def test_real_touch_toggle_on_off(self, mobile_page):
        chip = mobile_page.locator(".chips .chip").first
        self._real_tap(mobile_page, chip)
        assert get_chip_active(chip)
        self._real_tap(mobile_page, chip)
        assert not get_chip_active(chip)

    def test_real_touch_zero_active_after_off(self, mobile_page):
        chips = mobile_page.locator(".chips .chip")
        self._real_tap(mobile_page, chips.first)
        assert get_chip_active(chips.first)
        self._real_tap(mobile_page, chips.first)
        assert not get_chip_active(chips.first)
        active_count = chips.evaluate_all(
            "chips => chips.filter(c => c.dataset.active === 'true').length"
        )
        assert active_count == 0

    def test_real_touch_independent_chips(self, mobile_page):
        chips = mobile_page.locator(".chips .chip")
        self._real_tap(mobile_page, chips.nth(0))
        self._real_tap(mobile_page, chips.nth(1))
        assert get_chip_active(chips.nth(0)), "chip 0 should stay active"
        assert get_chip_active(chips.nth(1))
        self._real_tap(mobile_page, chips.nth(0))
        assert not get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1)), "chip 1 should stay active"

    def test_real_touch_deactivate_all_three(self, mobile_page):
        chips = mobile_page.locator(".chips .chip")
        for i in range(3):
            self._real_tap(mobile_page, chips.nth(i))
        for i in range(3):
            assert get_chip_active(chips.nth(i))
        for i in range(3):
            self._real_tap(mobile_page, chips.nth(i))
        for i in range(3):
            assert not get_chip_active(chips.nth(i))
        active_count = chips.evaluate_all(
            "chips => chips.filter(c => c.dataset.active === 'true').length"
        )
        assert active_count == 0


class TestSeedControls:
    def _set_seed(self, page, value):
        seed_input = page.locator("#seed")
        seed_input.evaluate(f"el => el.value = '{value}'")

    def _get_seed(self, page):
        return page.locator("#seed").evaluate("el => String(el.value)")

    def test_seed_plus_button(self, desktop_page):
        self._set_seed(desktop_page, "100")
        desktop_page.locator("#seedPlus").click()
        desktop_page.wait_for_timeout(500)
        assert self._get_seed(desktop_page) == "101"

    def test_seed_minus_button(self, desktop_page):
        self._set_seed(desktop_page, "100")
        desktop_page.locator("#seedMinus").click()
        desktop_page.wait_for_timeout(500)
        assert self._get_seed(desktop_page) == "99"

    def test_seed_minus_floors_at_zero(self, desktop_page):
        self._set_seed(desktop_page, "0")
        desktop_page.locator("#seedMinus").click()
        desktop_page.wait_for_timeout(500)
        assert self._get_seed(desktop_page) == "0"


class TestGeneration:
    def test_generate_returns_image(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'a red square'")
        desktop_page.locator("#seed").evaluate("el => el.value = '42'")
        desktop_page.locator("#genBtn").click()
        img = desktop_page.locator("#resultImg")
        img.wait_for(state="visible", timeout=30000)
        src = img.get_attribute("src")
        assert src and src.startswith("blob:")

    def test_metadata_displayed(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'a blue circle'")
        desktop_page.locator("#seed").evaluate("el => el.value = '42'")
        desktop_page.locator("#genBtn").click()
        meta = desktop_page.locator("#metaBox")
        meta.wait_for(state="visible", timeout=30000)
        assert desktop_page.locator("#mSeed").text_content() == "42"
        time_text = desktop_page.locator("#mTime").text_content()
        assert time_text.endswith("s")

    def test_share_url_displayed(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'test'")
        desktop_page.locator("#seed").evaluate("el => el.value = '1'")
        desktop_page.locator("#genBtn").click()
        share = desktop_page.locator("#shareBox")
        share.wait_for(state="visible", timeout=30000)
        url = desktop_page.locator("#shareUrl").evaluate("el => el.value")
        assert "prompt=test" in url
        assert "seed=1" in url

    def test_modifiers_included_in_prompt(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'a cat'")
        desktop_page.locator(".chip").first.click()
        desktop_page.locator("#seed").evaluate("el => el.value = '42'")
        desktop_page.locator("#genBtn").click()
        share = desktop_page.locator("#shareBox")
        share.wait_for(state="visible", timeout=30000)
        url = desktop_page.locator("#shareUrl").evaluate("el => el.value")
        assert "a+cat" in url or "a%20cat" in url
        first_chip_value = desktop_page.locator(".chip").first.get_attribute("data-v")
        assert first_chip_value.split(",")[0].strip() in url


class TestSourcesCRUD:
    """Test the sources CRUD API endpoints."""

    def test_sources_list(self):
        for s in requests.get(f"{BASE_URL}/sources", timeout=10).json():
            requests.delete(f"{BASE_URL}/sources/{s['id']}", timeout=10)
        resp = requests.get(f"{BASE_URL}/sources", timeout=10)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_source(self):
        resp = requests.post(
            f"{BASE_URL}/sources",
            json={"url": "https://example.com", "label": "Example"},
            timeout=10,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["url"] == "https://example.com"
        assert data["label"] == "Example"
        assert "id" in data
        # cleanup
        requests.delete(f"{BASE_URL}/sources/{data['id']}", timeout=10)

    def test_add_source_default_label(self):
        resp = requests.post(
            f"{BASE_URL}/sources",
            json={"url": "https://test.org"},
            timeout=10,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["label"] == "https://test.org"
        requests.delete(f"{BASE_URL}/sources/{data['id']}", timeout=10)

    def test_list_sources(self):
        r1 = requests.post(f"{BASE_URL}/sources", json={"url": "https://a.com"}, timeout=10)
        r2 = requests.post(f"{BASE_URL}/sources", json={"url": "https://b.com"}, timeout=10)
        resp = requests.get(f"{BASE_URL}/sources", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        urls = [s["url"] for s in data]
        assert "https://a.com" in urls
        assert "https://b.com" in urls
        # cleanup
        requests.delete(f"{BASE_URL}/sources/{r1.json()['id']}", timeout=10)
        requests.delete(f"{BASE_URL}/sources/{r2.json()['id']}", timeout=10)

    def test_delete_source(self):
        r = requests.post(f"{BASE_URL}/sources", json={"url": "https://del.com"}, timeout=10)
        sid = r.json()["id"]
        resp = requests.delete(f"{BASE_URL}/sources/{sid}", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        # verify gone
        sources = requests.get(f"{BASE_URL}/sources", timeout=10).json()
        assert all(s["id"] != sid for s in sources)

    def test_delete_nonexistent_source(self):
        resp = requests.delete(f"{BASE_URL}/sources/nonexistent-id", timeout=10)
        assert resp.status_code == 404

    def test_save_prompt_as_source(self):
        resp = requests.post(
            f"{BASE_URL}/sources/from-prompt",
            json={"prompt": "a beautiful sunset", "chips": ["Cinematic"]},
            timeout=30,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert len(data["label"]) > 0
        assert data["label"] != "Saved prompt"
        assert data["url"] == "a beautiful sunset"
        assert data["chips"] == ["Cinematic"]
        assert "id" in data
        requests.delete(f"{BASE_URL}/sources/{data['id']}", timeout=10)


class TestModelRendering:
    """Integration tests that load each model, generate an image, and verify
    the response is a valid, non-trivial PNG with the correct X-Model header."""

    MODELS = ["sd-turbo", "dreamshaper-8", "lcm-dreamshaper"]
    DEFAULT_PROMPT = "a photo of a cat"

    @classmethod
    def _load_model(cls, model_key):
        resp = requests.post(
            f"{BASE_URL}/load_model",
            json={"model": model_key},
            timeout=120,
        )
        assert resp.status_code == 200, f"Failed to load {model_key}: {resp.text}"
        data = resp.json()
        assert data["status"] in ("loaded", "already_loaded")

    @classmethod
    def _generate(cls, prompt, seed=42):
        resp = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": prompt, "seed": seed},
            timeout=120,
        )
        return resp

    @staticmethod
    def _check_image_nontrivial(png_bytes, min_std=1.0):
        """Decode PNG and check that pixel values have non-trivial variance."""
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        std = arr.std()
        assert std > min_std, (
            f"Image appears to be a single color (std={std:.2f}). "
            "Expected non-trivial content."
        )

    def test_sd_turbo(self):
        self._load_model("sd-turbo")
        resp = self._generate(self.DEFAULT_PROMPT, seed=42)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["x-model"] == "sd-turbo"
        self._check_image_nontrivial(resp.content)

    def test_dreamshaper_8(self):
        self._load_model("dreamshaper-8")
        resp = self._generate(self.DEFAULT_PROMPT, seed=42)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["x-model"] == "dreamshaper-8"
        self._check_image_nontrivial(resp.content)

    def test_lcm_dreamshaper(self):
        self._load_model("lcm-dreamshaper")
        resp = self._generate(self.DEFAULT_PROMPT, seed=42)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["x-model"] == "lcm-dreamshaper"
        self._check_image_nontrivial(resp.content)

    def test_restore_sd_turbo(self):
        """After testing all models, switch back to sd-turbo."""
        self._load_model("sd-turbo")
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["active"] == "sd-turbo"


class TestGenerationMetadata:
    """Verify all generation key-value pairs are returned and displayed."""

    def test_all_headers_present(self):
        resp = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a red car", "negative_prompt": "blurry", "seed": 77, "guidance_scale": 1.0},
            timeout=30,
        )
        assert resp.status_code == 200
        for hdr in ["x-prompt", "x-negative-prompt", "x-seed", "x-steps",
                     "x-guidance-scale", "x-model", "x-model-label",
                     "x-cfg-type", "x-width", "x-height", "x-generation-time"]:
            assert hdr in resp.headers, f"Missing header: {hdr}"
        assert resp.headers["x-prompt"] == "a red car"
        assert resp.headers["x-negative-prompt"] == "blurry"
        assert resp.headers["x-seed"] == "77"
        assert resp.headers["x-guidance-scale"] == "1.0"

    def test_empty_negative_prompt_shows_none(self):
        resp = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a dog", "negative_prompt": "", "seed": 1},
            timeout=30,
        )
        assert resp.status_code == 200
        assert resp.headers["x-negative-prompt"] == "(none)"

    def test_meta_card_shows_all_fields(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'sunset beach'")
        desktop_page.locator("#seed").evaluate("el => el.value = '99'")
        desktop_page.locator("#genBtn").click()
        meta = desktop_page.locator("#metaBox")
        meta.wait_for(state="visible", timeout=30000)
        assert "sunset beach" in desktop_page.locator("#mPrompt").text_content()
        assert desktop_page.locator("#mSeed").text_content() == "99"
        assert desktop_page.locator("#mCfgType").text_content() in ("none", "full", "initialize")
        assert desktop_page.locator("#mSteps").text_content() == "50"
        assert desktop_page.locator("#mGuidance").text_content()
        assert desktop_page.locator("#mTime").text_content().endswith("s")
        assert desktop_page.locator("#mModel").text_content()

    def test_meta_card_shows_negative_prompt(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'a cat'")
        desktop_page.locator("#seed").evaluate("el => el.value = '42'")
        neg = desktop_page.locator("#negPrompt")
        neg.evaluate("el => el.value = 'ugly watermark'")
        desktop_page.locator("#genBtn").click()
        meta = desktop_page.locator("#metaBox")
        meta.wait_for(state="visible", timeout=30000)
        neg_text = desktop_page.locator("#mNeg").text_content()
        assert "ugly watermark" in neg_text


class TestNegativePrompt:
    """Verify negative prompt is saved, applied, and round-trips through share URL."""

    def test_negative_prompt_changes_output(self):
        """Same prompt+seed with different negative prompts should produce different images."""
        params_base = {"prompt": "a forest", "negative_prompt": "", "seed": 123, "guidance_scale": 1.2}
        params_neg = {**params_base, "negative_prompt": "dark gloomy"}
        # Need a model that supports CFG for this to matter
        requests.post(f"{BASE_URL}/load_model", json={"model": "dreamshaper-8"}, timeout=120)
        r1 = requests.get(f"{BASE_URL}/generate", params=params_base, timeout=60)
        r2 = requests.get(f"{BASE_URL}/generate", params=params_neg, timeout=60)
        assert r1.status_code == 200
        assert r2.status_code == 200
        a = np.array(Image.open(io.BytesIO(r1.content)), dtype=np.float32)
        b = np.array(Image.open(io.BytesIO(r2.content)), dtype=np.float32)
        diff = np.abs(a - b).mean()
        assert diff > 1.0, f"Negative prompt had no effect (mean diff={diff:.2f})"

    def test_cfg_type_matches_model(self):
        """SD-turbo should report cfg_type=none, DreamShaper should report full."""
        requests.post(f"{BASE_URL}/load_model", json={"model": "sd-turbo"}, timeout=120)
        r = requests.get(f"{BASE_URL}/generate", params={"prompt": "test", "seed": 1}, timeout=30)
        assert r.headers["x-cfg-type"] == "none"

        requests.post(f"{BASE_URL}/load_model", json={"model": "dreamshaper-8"}, timeout=120)
        r = requests.get(f"{BASE_URL}/generate", params={"prompt": "test", "seed": 1, "guidance_scale": 1.2}, timeout=60)
        assert r.headers["x-cfg-type"] == "full"

    def test_guidance_1_with_cfg_full_no_crash(self):
        """cfg_type=full with guidance_scale=1.0 should not crash (regression)."""
        requests.post(f"{BASE_URL}/load_model", json={"model": "dreamshaper-8"}, timeout=120)
        r = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a tree", "negative_prompt": "ugly", "seed": 42, "guidance_scale": 1.0},
            timeout=60,
        )
        assert r.status_code == 200
        assert r.headers["x-cfg-type"] == "full"


class TestShareURLRoundTrip:
    """Share URL should encode all params including negative prompt and restore them."""

    def test_share_url_includes_negative_prompt(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = 'a horse'")
        desktop_page.locator("#negPrompt").evaluate("el => el.value = 'cartoon'")
        desktop_page.locator("#seed").evaluate("el => el.value = '55'")
        desktop_page.locator("#genBtn").click()
        share = desktop_page.locator("#shareBox")
        share.wait_for(state="visible", timeout=30000)
        url = desktop_page.locator("#shareUrl").evaluate("el => el.value")
        assert "negative_prompt=cartoon" in url
        assert "seed=55" in url

    def test_share_url_restores_negative_prompt(self, desktop_page):
        desktop_page.goto(f"{BASE_URL}/?prompt=test+horse&negative_prompt=blurry+ugly&seed=88&guidance_scale=1.0")
        desktop_page.wait_for_load_state("domcontentloaded")
        wait_for_shoelace(desktop_page)
        neg_val = desktop_page.locator("#negPrompt").evaluate("el => el.value")
        assert neg_val == "blurry ugly"
        seed_val = desktop_page.locator("#seed").evaluate("el => String(el.value)")
        assert seed_val == "88"
        guidance_val = desktop_page.locator("#guidance").evaluate("el => String(el.value)")
        assert float(guidance_val) == 1.0


class TestModelGuidanceDefault:
    """Switching models should update the guidance slider to the model's default."""

    def test_load_model_returns_guidance(self):
        resp = requests.post(
            f"{BASE_URL}/load_model",
            json={"model": "dreamshaper-8"},
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guidance"] == 1.2

    def test_load_sd_turbo_returns_guidance_1(self):
        resp = requests.post(
            f"{BASE_URL}/load_model",
            json={"model": "sd-turbo"},
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["guidance"] == 1.0

    def test_models_endpoint_includes_guidance(self):
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert models["sd-turbo"]["guidance"] == 1.0
        assert models["dreamshaper-8"]["guidance"] == 1.2


class TestIPAdapter:
    """Test IP-Adapter upload, generation with reference, clear, and status."""

    @staticmethod
    def _make_test_image(color=(128, 64, 32)):
        img = Image.new("RGB", (256, 256), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_models_reports_ip_adapter_status(self):
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "has_ip_adapter" in data
        assert "has_ip_reference" in data

    def test_ip_adapter_upload_requires_compatible_model(self):
        requests.post(f"{BASE_URL}/load_model", json={"model": "sd-turbo"}, timeout=120)
        buf = self._make_test_image()
        resp = requests.post(
            f"{BASE_URL}/ip-adapter/upload",
            files={"file": ("ref.png", buf, "image/png")},
            timeout=30,
        )
        assert resp.status_code == 400
        assert "not loaded" in resp.json()["detail"]

    def test_ip_adapter_full_lifecycle(self):
        requests.post(f"{BASE_URL}/load_model", json={"model": "dreamshaper-8"}, timeout=120)

        # Verify IP-Adapter available
        models = requests.get(f"{BASE_URL}/models", timeout=10).json()
        assert models["has_ip_adapter"] is True
        assert models["has_ip_reference"] is False

        # Generate without reference
        r1 = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a portrait", "seed": 42, "guidance_scale": 1.2},
            timeout=60,
        )
        assert r1.status_code == 200
        assert r1.headers["x-ip-adapter"] == "off"

        # Upload reference
        buf = self._make_test_image((200, 100, 50))
        resp = requests.post(
            f"{BASE_URL}/ip-adapter/upload?scale=0.6",
            files={"file": ("ref.png", buf, "image/png")},
            timeout=30,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify reference active
        models = requests.get(f"{BASE_URL}/models", timeout=10).json()
        assert models["has_ip_reference"] is True

        # Generate with reference
        r2 = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a portrait", "seed": 42, "guidance_scale": 1.2},
            timeout=60,
        )
        assert r2.status_code == 200
        assert r2.headers["x-ip-adapter"] == "active"

        # Output should differ from no-reference
        a = np.array(Image.open(io.BytesIO(r1.content)), dtype=np.float32)
        b = np.array(Image.open(io.BytesIO(r2.content)), dtype=np.float32)
        diff = np.abs(a - b).mean()
        assert diff > 1.0, f"IP-Adapter had no effect (mean diff={diff:.2f})"

        # Clear reference
        resp = requests.post(f"{BASE_URL}/ip-adapter/clear", timeout=10)
        assert resp.status_code == 200
        models = requests.get(f"{BASE_URL}/models", timeout=10).json()
        assert models["has_ip_reference"] is False

        # Generate after clear should match original (zero embeds)
        r3 = requests.get(
            f"{BASE_URL}/generate",
            params={"prompt": "a portrait", "seed": 42, "guidance_scale": 1.2},
            timeout=60,
        )
        assert r3.status_code == 200
        assert r3.headers["x-ip-adapter"] == "off"


class TestInputValidation:
    def test_empty_prompt_rejected(self, desktop_page):
        desktop_page.locator("#prompt").evaluate("el => el.value = ''")
        desktop_page.locator("#seed").evaluate("el => el.value = '42'")
        desktop_page.locator("#genBtn").click()
        desktop_page.wait_for_function(
            "() => document.querySelector('sl-alert[variant=\"danger\"]') !== null",
            timeout=10000,
        )
        text = desktop_page.locator("sl-alert[variant='danger']").first.text_content()
        assert "1-1000" in text or "Prompt must be" in text


class TestPromptEnhancement:
    """Test the POST /enhance endpoint for prompt enhancement via Claude API."""

    def test_enhance_returns_expected_fields(self):
        """POST /enhance with a valid prompt returns enhanced_prompt, negative_prompt, notes."""
        resp = requests.post(
            f"{BASE_URL}/enhance",
            json={"prompt": "a pack of three coyotes threatens the viewer"},
            timeout=30,
        )
        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "enhanced_prompt" in data
        assert "negative_prompt" in data
        assert "notes" in data
        assert "original_prompt" in data
        assert len(data["enhanced_prompt"]) > 0
        assert len(data["negative_prompt"]) > 0
        assert len(data["notes"]) > 0
        assert data["original_prompt"] == "a pack of three coyotes threatens the viewer"

    def test_enhance_incorporates_context(self):
        """Context fields should influence the enhanced prompt."""
        resp = requests.post(
            f"{BASE_URL}/enhance",
            json={
                "prompt": "a castle on a hill",
                "context": {
                    "mood": "menacing, eerie",
                    "genre": "dark fantasy",
                    "location": "mountain peak",
                    "time_of_day": "twilight",
                    "environment": "stormy, lightning",
                    "camera": "low angle, wide",
                    "medium": "digital painting",
                },
            },
            timeout=30,
        )
        assert resp.status_code == 200, f"Expected 200 but got {resp.status_code}: {resp.text}"
        data = resp.json()
        enhanced = data["enhanced_prompt"].lower()
        # The enhanced prompt should reflect at least some context fields
        context_hits = 0
        for term in ["dark", "fantasy", "twilight", "storm", "lightning", "low angle",
                      "digital", "painting", "mountain", "menac", "eerie"]:
            if term in enhanced:
                context_hits += 1
        assert context_hits >= 2, (
            f"Expected at least 2 context terms in enhanced prompt, found {context_hits}. "
            f"Prompt: {data['enhanced_prompt']}"
        )

    def test_enhance_empty_prompt_returns_400(self):
        """An empty prompt should return HTTP 400."""
        resp = requests.post(
            f"{BASE_URL}/enhance",
            json={"prompt": ""},
            timeout=10,
        )
        assert resp.status_code == 400

        resp2 = requests.post(
            f"{BASE_URL}/enhance",
            json={"prompt": "   "},
            timeout=10,
        )
        assert resp2.status_code == 400

    def test_enhance_then_generate_produces_image(self):
        """Full flow: enhance a prompt, then use the result to generate a valid image."""
        # Step 1: Enhance
        enhance_resp = requests.post(
            f"{BASE_URL}/enhance",
            json={
                "prompt": "a wolf howling at the moon",
                "context": {"mood": "mysterious", "time_of_day": "night"},
            },
            timeout=30,
        )
        assert enhance_resp.status_code == 200, (
            f"Enhance failed: {enhance_resp.status_code} {enhance_resp.text}"
        )
        enhanced_data = enhance_resp.json()
        enhanced_prompt = enhanced_data["enhanced_prompt"]
        negative_prompt = enhanced_data["negative_prompt"]

        # Step 2: Generate with the enhanced prompt
        gen_resp = requests.get(
            f"{BASE_URL}/generate",
            params={
                "prompt": enhanced_prompt,
                "negative_prompt": negative_prompt,
                "seed": 42,
                "guidance_scale": 1.0,
            },
            timeout=60,
        )
        assert gen_resp.status_code == 200, (
            f"Generate failed: {gen_resp.status_code} {gen_resp.text}"
        )
        assert gen_resp.headers["content-type"] == "image/png"

        # Verify it's a valid, non-trivial image
        img = Image.open(io.BytesIO(gen_resp.content)).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        assert arr.std() > 1.0, "Generated image appears blank"


class TestComposeArea:
    """Tests for the compose area: active pills, #autocomplete, chip↔pill sync."""

    def test_active_pills_container_exists(self, desktop_page):
        pills = desktop_page.locator("#activePills")
        expect(pills).to_be_attached()
        expect(pills).to_be_hidden()

    def test_chip_toggle_creates_pill(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        chip.click()
        assert get_chip_active(chip)
        pills = desktop_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(1)
        label = chip.text_content().strip()
        assert label in pills.first.text_content()

    def test_pill_x_deactivates_chip(self, desktop_page):
        chip = desktop_page.locator(".chip").first
        chip.click()
        assert get_chip_active(chip)
        pills = desktop_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(1)
        pills.first.locator(".pill-x").click()
        assert not get_chip_active(chip)
        expect(desktop_page.locator("#activePills .active-pill")).to_have_count(0)

    def test_multiple_chips_create_multiple_pills(self, desktop_page):
        chips = desktop_page.locator(".chip")
        chips.nth(0).click()
        chips.nth(1).click()
        pills = desktop_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(2)
        chips.nth(0).click()
        expect(desktop_page.locator("#activePills .active-pill")).to_have_count(1)

    def test_autocomplete_opens_on_hash(self, desktop_page):
        prompt = desktop_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.focus()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#photo';
            ta.selectionStart = ta.selectionEnd = 6;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = desktop_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        items = desktop_page.locator(".autocomplete-item")
        assert items.count() >= 1
        assert "Photorealistic" in items.first.text_content()

    def test_autocomplete_select_activates_chip(self, desktop_page):
        prompt = desktop_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.focus()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#cine';
            ta.selectionStart = ta.selectionEnd = 5;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = desktop_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        desktop_page.locator(".autocomplete-item").first.click()
        cinematic_chip = desktop_page.locator(".chip", has_text="Cinematic")
        assert get_chip_active(cinematic_chip)
        pills = desktop_page.locator("#activePills .active-pill")
        assert any("Cinematic" in pills.nth(i).text_content() for i in range(pills.count()))
        cinematic_chip.click()

    def test_autocomplete_closes_on_escape(self, desktop_page):
        prompt = desktop_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.focus()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#gold';
            ta.selectionStart = ta.selectionEnd = 5;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = desktop_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        desktop_page.evaluate("""() => {
            const el = document.getElementById('prompt');
            el.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        }""")
        assert "open" not in (ac_list.get_attribute("class") or "")

    def test_pills_empty_when_no_chips_active(self, desktop_page):
        desktop_page.evaluate("document.querySelectorAll('.chip').forEach(c => c.dataset.active = 'false')")
        desktop_page.evaluate("document.querySelector('.chip').click()")
        desktop_page.evaluate("document.querySelector('.chip').click()")
        pills = desktop_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(0)


class TestComposeAreaMobile:
    """Mobile touch-specific tests for compose area pills and autocomplete."""

    def test_chip_tap_creates_pill(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        chip.tap()
        assert get_chip_active(chip)
        pills = mobile_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(1)
        label = chip.text_content().strip()
        assert label in pills.first.text_content()
        chip.tap()

    def test_pill_x_tap_deactivates_chip(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        chip.tap()
        assert get_chip_active(chip)
        pill_x = mobile_page.locator("#activePills .active-pill .pill-x").first
        pill_x.tap()
        assert not get_chip_active(chip)
        expect(mobile_page.locator("#activePills .active-pill")).to_have_count(0)

    def test_pill_touch_target_min_size(self, mobile_page):
        chip = mobile_page.locator(".chip").first
        chip.tap()
        pill = mobile_page.locator("#activePills .active-pill").first
        box = pill.bounding_box()
        assert box["height"] >= 28, f"Pill height {box['height']}px is too small for touch"
        pill_x = pill.locator(".pill-x")
        x_box = pill_x.bounding_box()
        assert x_box["width"] >= 24, f"Pill X width {x_box['width']}px is too small for touch"
        assert x_box["height"] >= 24, f"Pill X height {x_box['height']}px is too small for touch"
        chip.tap()

    def test_autocomplete_item_touch_target(self, mobile_page):
        prompt = mobile_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.tap()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#photo';
            ta.selectionStart = ta.selectionEnd = 6;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = mobile_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        item = mobile_page.locator(".autocomplete-item").first
        box = item.bounding_box()
        assert box["height"] >= 40, f"Autocomplete item height {box['height']}px is too small for touch"
        mobile_page.evaluate("""() => {
            document.getElementById('acList').classList.remove('open');
        }""")

    def test_autocomplete_drops_below_on_mobile(self, mobile_page):
        prompt = mobile_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.tap()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#dram';
            ta.selectionStart = ta.selectionEnd = 5;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = mobile_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        ac_box = ac_list.bounding_box()
        prompt_box = prompt.bounding_box()
        assert ac_box["y"] >= prompt_box["y"], "Autocomplete should drop below textarea on mobile"
        mobile_page.evaluate("""() => {
            document.getElementById('acList').classList.remove('open');
        }""")

    def test_autocomplete_tap_select_activates_chip(self, mobile_page):
        prompt = mobile_page.locator("#prompt")
        prompt.evaluate("el => el.value = ''")
        prompt.tap()
        prompt.evaluate("""el => {
            const ta = el.shadowRoot.querySelector('textarea');
            ta.value = '#neon';
            ta.selectionStart = ta.selectionEnd = 5;
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new CustomEvent('sl-input', {bubbles: true}));
        }""")
        ac_list = mobile_page.locator("#acList")
        assert "open" in (ac_list.get_attribute("class") or "")
        mobile_page.locator(".autocomplete-item").first.tap()
        neon_chip = mobile_page.locator(".chip", has_text="Neon")
        assert get_chip_active(neon_chip)
        pills = mobile_page.locator("#activePills .active-pill")
        assert any("Neon" in pills.nth(i).text_content() for i in range(pills.count()))
        mobile_page.evaluate("""() => {
            document.querySelectorAll('.chip').forEach(c => c.dataset.active = 'false');
            document.getElementById('activePills').innerHTML = '';
        }""")

    def test_multiple_pills_wrap_on_narrow_screen(self, mobile_page):
        chips = mobile_page.locator(".chip")
        for i in range(4):
            chips.nth(i).tap()
        pills = mobile_page.locator("#activePills .active-pill")
        expect(pills).to_have_count(4)
        container = mobile_page.locator("#activePills")
        box = container.bounding_box()
        assert box["width"] <= 420, "Pills container should fit mobile width"
        for i in range(4):
            chips.nth(i).tap()


class TestSceneCRUD:
    """Full CRUD lifecycle for scene types and presets."""

    CUSTOM_SCENE = {
        "label": "Video Production",
        "categories": [
            {"name": "Style", "presets": [
                {"label": "Documentary", "value": "documentary, handheld, natural"},
                {"label": "Music Video", "value": "music video, stylized, fast cuts"},
            ]},
            {"name": "Pacing", "presets": [
                {"label": "Slow Motion", "value": "slow motion, 120fps, dreamy"},
            ]},
        ],
    }

    def test_list_scenes(self):
        resp = requests.get(f"{BASE_URL}/scenes", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "image" in data
        assert "music" in data
        assert "sound" in data
        assert data["image"]["category_count"] == 3
        assert data["image"]["preset_count"] == 27

    def test_get_scene(self):
        resp = requests.get(f"{BASE_URL}/scenes/image", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "image"
        assert data["label"] == "Image Generation"
        assert len(data["categories"]) == 3

    def test_get_scene_not_found(self):
        resp = requests.get(f"{BASE_URL}/scenes/nonexistent", timeout=10)
        assert resp.status_code == 404

    def test_put_create_scene(self):
        resp = requests.put(
            f"{BASE_URL}/scenes/test-video",
            json=self.CUSTOM_SCENE, timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "test-video"
        assert data["label"] == "Video Production"
        assert len(data["categories"]) == 2
        scenes = requests.get(f"{BASE_URL}/scenes", timeout=10).json()
        assert "test-video" in scenes
        requests.delete(f"{BASE_URL}/scenes/test-video", timeout=10)

    def test_put_update_scene(self):
        requests.put(f"{BASE_URL}/scenes/test-update", json=self.CUSTOM_SCENE, timeout=10)
        updated = {
            "label": "Updated Video",
            "categories": [
                {"name": "NewCat", "presets": [{"label": "NewPreset", "value": "new, value"}]},
            ],
        }
        resp = requests.put(f"{BASE_URL}/scenes/test-update", json=updated, timeout=10)
        assert resp.status_code == 200
        assert resp.json()["label"] == "Updated Video"
        requests.delete(f"{BASE_URL}/scenes/test-update", timeout=10)

    def test_patch_add_category(self):
        requests.put(f"{BASE_URL}/scenes/test-patch", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.patch(
            f"{BASE_URL}/scenes/test-patch",
            json={"add_category": {"name": "Transition", "presets": [{"label": "Crossfade", "value": "crossfade, dissolve"}]}},
            timeout=10,
        )
        assert resp.status_code == 200
        cats = [c["name"] for c in resp.json()["categories"]]
        assert "Transition" in cats
        requests.delete(f"{BASE_URL}/scenes/test-patch", timeout=10)

    def test_patch_remove_category(self):
        requests.put(f"{BASE_URL}/scenes/test-patch2", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.patch(
            f"{BASE_URL}/scenes/test-patch2",
            json={"remove_category": "Pacing"}, timeout=10,
        )
        cats = [c["name"] for c in resp.json()["categories"]]
        assert "Pacing" not in cats
        assert "Style" in cats
        requests.delete(f"{BASE_URL}/scenes/test-patch2", timeout=10)

    def test_patch_add_presets(self):
        requests.put(f"{BASE_URL}/scenes/test-patch3", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.patch(
            f"{BASE_URL}/scenes/test-patch3",
            json={"add_presets": {"category": "Style", "presets": [{"label": "Vlog", "value": "vlog, casual"}]}},
            timeout=10,
        )
        style_cat = [c for c in resp.json()["categories"] if c["name"] == "Style"][0]
        assert "Vlog" in [p["label"] for p in style_cat["presets"]]
        requests.delete(f"{BASE_URL}/scenes/test-patch3", timeout=10)

    def test_patch_remove_presets(self):
        requests.put(f"{BASE_URL}/scenes/test-patch4", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.patch(
            f"{BASE_URL}/scenes/test-patch4",
            json={"remove_presets": {"category": "Style", "labels": ["Documentary"]}},
            timeout=10,
        )
        style_cat = [c for c in resp.json()["categories"] if c["name"] == "Style"][0]
        labels = [p["label"] for p in style_cat["presets"]]
        assert "Documentary" not in labels
        assert "Music Video" in labels
        requests.delete(f"{BASE_URL}/scenes/test-patch4", timeout=10)

    def test_delete_scene(self):
        requests.put(f"{BASE_URL}/scenes/test-del", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.delete(f"{BASE_URL}/scenes/test-del", timeout=10)
        assert resp.status_code == 200
        assert requests.get(f"{BASE_URL}/scenes/test-del", timeout=10).status_code == 404

    def test_delete_not_found(self):
        assert requests.delete(f"{BASE_URL}/scenes/nonexistent", timeout=10).status_code == 404

    def test_put_invalid_key(self):
        resp = requests.put(f"{BASE_URL}/scenes/INVALID!", json=self.CUSTOM_SCENE, timeout=10)
        assert resp.status_code == 400

    def test_put_duplicate_preset_labels(self):
        bad = {"label": "Bad", "categories": [
            {"name": "Cat", "presets": [{"label": "Same", "value": "a"}, {"label": "Same", "value": "b"}]}
        ]}
        assert requests.put(f"{BASE_URL}/scenes/test-bad", json=bad, timeout=10).status_code == 400

    def test_patch_duplicate_category(self):
        requests.put(f"{BASE_URL}/scenes/test-dup", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.patch(
            f"{BASE_URL}/scenes/test-dup",
            json={"add_category": {"name": "Style", "presets": [{"label": "X", "value": "x"}]}},
            timeout=10,
        )
        assert resp.status_code == 409
        requests.delete(f"{BASE_URL}/scenes/test-dup", timeout=10)

    def test_presets_endpoint_serves_custom(self):
        requests.put(f"{BASE_URL}/scenes/test-presets", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.get(f"{BASE_URL}/presets?scene=test-presets", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["label"] == "Video Production"
        requests.delete(f"{BASE_URL}/scenes/test-presets", timeout=10)

    def test_enhance_with_custom_scene(self):
        requests.put(f"{BASE_URL}/scenes/test-enh", json=self.CUSTOM_SCENE, timeout=10)
        resp = requests.post(
            f"{BASE_URL}/enhance",
            json={"prompt": "a cinematic drone shot over mountains", "scene": "test-enh"},
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "enhanced_prompt" in data
        assert "suggested_chips" in data
        requests.delete(f"{BASE_URL}/scenes/test-enh", timeout=10)
