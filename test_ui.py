"""Playwright integration tests for the StreamDiffusion web app (Shoelace UI).

Covers chip toggle behavior (including mobile touch), seed controls,
generation flow, and input validation.

Run:  .venv/bin/python -m pytest test_ui.py -v
"""

import pytest
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
        section_chips = mobile_page.locator("#styleChips .chip")
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
        chips = mobile_page.locator("#styleChips .chip")
        chips.nth(0).tap()
        assert get_chip_active(chips.nth(0))
        chips.nth(1).tap()
        assert get_chip_active(chips.nth(0)), "first chip lost its active state"
        assert get_chip_active(chips.nth(1))

    def test_tap_sequence_activates_then_deactivates_each(self, mobile_page):
        """Activate 3 chips, deactivate each individually, confirm zero remain."""
        chips = mobile_page.locator("#styleChips .chip")
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
        chip = mobile_page.locator("#styleChips .chip").first
        self._real_tap(mobile_page, chip)
        assert get_chip_active(chip)
        self._real_tap(mobile_page, chip)
        assert not get_chip_active(chip)

    def test_real_touch_zero_active_after_off(self, mobile_page):
        chips = mobile_page.locator("#styleChips .chip")
        self._real_tap(mobile_page, chips.first)
        assert get_chip_active(chips.first)
        self._real_tap(mobile_page, chips.first)
        assert not get_chip_active(chips.first)
        active_count = chips.evaluate_all(
            "chips => chips.filter(c => c.dataset.active === 'true').length"
        )
        assert active_count == 0

    def test_real_touch_independent_chips(self, mobile_page):
        chips = mobile_page.locator("#styleChips .chip")
        self._real_tap(mobile_page, chips.nth(0))
        self._real_tap(mobile_page, chips.nth(1))
        assert get_chip_active(chips.nth(0)), "chip 0 should stay active"
        assert get_chip_active(chips.nth(1))
        self._real_tap(mobile_page, chips.nth(0))
        assert not get_chip_active(chips.nth(0))
        assert get_chip_active(chips.nth(1)), "chip 1 should stay active"

    def test_real_touch_deactivate_all_three(self, mobile_page):
        chips = mobile_page.locator("#styleChips .chip")
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
