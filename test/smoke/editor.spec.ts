import { browser } from "@wdio/globals";
import assert from "node:assert/strict";
import type {} from "webdriverio";
import { describe, it } from "mocha";

interface NavigationResult {
  readonly fps: number;
  readonly frames: number;
  readonly elapsedMs: number;
  readonly error: string | null;
}

interface TauriWindowApi {
  getCurrentWindow(): {
    setFocus(): Promise<void>;
  };
}

interface FocusResult {
  readonly error: string | null;
}

interface EditFeedbackResult {
  readonly dirty: boolean;
  readonly feedbackMs: number;
  readonly responsiveMs: number;
  readonly error: string | null;
}

describe("VoxelInk native shell", () => {
  it("opens the editor and navigates the prepared reference volume at 30 fps", async () => {
    const app = browser.$("[data-testid='voxel-editor']");
    const open = browser.$("button=Open volume");
    const brush = browser.$("button=Brush");
    const axial = browser.$("[aria-label='Axial slice']");
    const ready = browser.$("span=3D preview ready");
    assert.equal(await app.isDisplayed(), true);
    assert.equal(await open.isDisplayed(), true);
    assert.equal(await brush.isDisplayed(), true);
    assert.equal(await axial.isDisplayed(), true);
    await ready.waitForDisplayed({ timeout: 120_000 });

    const focusResult = await browser.executeAsync<FocusResult, []>((done) => {
      const tauri = (window as unknown as {
        __TAURI__?: { window?: TauriWindowApi };
      }).__TAURI__;
      if (!tauri?.window) {
        done({ error: "The Tauri window API is unavailable" });
        return;
      }
      tauri.window.getCurrentWindow().setFocus().then(
        () => done({ error: null }),
        (error: unknown) => done({ error: String(error) }),
      );
    });
    assert.equal(focusResult.error, null);
    await browser.waitUntil(
      () => browser.execute(() => document.hasFocus()),
      {
        timeout: 5_000,
        timeoutMsg: "The native editor window did not receive focus for frame measurement",
      },
    );

    const result = await browser.executeAsync<NavigationResult, []>((done) => {
      const canvas = document.querySelector<HTMLCanvasElement>("[aria-label='3D volume view']");
      if (!canvas) {
        done({ fps: 0, frames: 0, elapsedMs: 0, error: "3D canvas is missing" });
        return;
      }
      const navigationEvent = "voxelink:frame-rendered";
      const warmupFrameCount = 5;
      let warmupFrames = 0;
      const renderTimestamps: number[] = [];
      let finished = false;
      const finish = (value: NavigationResult): void => {
        if (finished) return;
        finished = true;
        window.clearTimeout(watchdog);
        canvas.removeEventListener(navigationEvent, onRendered);
        done(value);
      };
      const watchdog = window.setTimeout(() => {
        const first = renderTimestamps[0];
        finish({
          fps: 0,
          frames: renderTimestamps.length,
          elapsedMs: first === undefined ? 0 : performance.now() - first,
          error: `Navigation rendering stalled (visibility=${document.visibilityState}, focus=${document.hasFocus()})`,
        });
      }, 10_000);

      const navigate = (): void => {
        canvas.dispatchEvent(new WheelEvent("wheel", {
          bubbles: true,
          cancelable: true,
          deltaY: (warmupFrames + renderTimestamps.length) % 2 === 0 ? 6 : -6,
        }));
      };
      const onRendered = (): void => {
        if (warmupFrames < warmupFrameCount) {
          warmupFrames += 1;
          navigate();
          return;
        }
        renderTimestamps.push(performance.now());
        const first = renderTimestamps[0];
        const last = renderTimestamps.at(-1);
        if (first === undefined || last === undefined || last - first < 2_000) {
          navigate();
          return;
        }
        const elapsedMs = last - first;
        finish({
          fps: (renderTimestamps.length - 1) * 1_000 / elapsedMs,
          frames: renderTimestamps.length,
          elapsedMs,
          error: null,
        });
      };
      canvas.addEventListener(navigationEvent, onRendered);
      navigate();
    });

    assert.equal(result.error, null);
    console.info(
      `VoxelInk native navigation benchmark: ${result.fps.toFixed(2)} fps across ${result.frames} completed renders in ${result.elapsedMs.toFixed(1)} ms`,
    );
    assert.ok(result.fps >= 30, `Expected at least 30 fps, observed ${result.fps.toFixed(2)} fps`);
  }).timeout(180_000);

  it("keeps brush feedback responsive while the 3D mesh catches up", async () => {
    const axialCanvas = browser.$("[aria-label='Axial slice']");
    const brushButton = browser.$("button=Brush");
    const ready = browser.$("span=3D preview ready");
    await axialCanvas.waitForDisplayed({ timeout: 30_000 });
    await brushButton.waitForEnabled({ timeout: 30_000 });
    await ready.waitForDisplayed({ timeout: 120_000 });

    const result = await browser.executeAsync<EditFeedbackResult, []>((done) => {
      const brush = [...document.querySelectorAll<HTMLButtonElement>("button")]
        .find((button) => button.textContent === "Brush");
      const axial = document.querySelector<HTMLCanvasElement>("[aria-label='Axial slice']");
      const save = [...document.querySelectorAll<HTMLButtonElement>("button")]
        .find((button) => button.textContent === "Save");
      if (!brush || !axial || !save) {
        done({ dirty: false, feedbackMs: 0, responsiveMs: 0, error: "Brush controls are missing" });
        return;
      }

      brush.click();
      axial.focus();
      window.setTimeout(() => {
        const started = performance.now();
        let feedbackMs = Number.POSITIVE_INFINITY;
        window.setTimeout(() => { feedbackMs = performance.now() - started; }, 0);
        axial.dispatchEvent(new KeyboardEvent("keydown", {
          key: " ",
          bubbles: true,
          cancelable: true,
        }));
        window.setTimeout(() => {
          done({
            dirty: !save.disabled,
            feedbackMs,
            responsiveMs: performance.now() - started,
            error: null,
          });
        }, 250);
      }, 0);
    });

    assert.equal(result.error, null);
    assert.equal(result.dirty, true, "The brush edit did not update the document");
    assert.ok(result.feedbackMs < 100, `Expected edit feedback under 100 ms, observed ${result.feedbackMs.toFixed(1)} ms`);
    assert.ok(result.responsiveMs < 1_000, `The UI stopped responding after the brush edit for ${result.responsiveMs.toFixed(1)} ms`);
  }).timeout(30_000);
});
