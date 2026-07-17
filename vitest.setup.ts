import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
});

class TestResizeObserver implements ResizeObserver {
  readonly root = null;
  readonly rootMargin = "0px";
  readonly thresholds = [];
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe(target: Element): void {
    this.callback([{ target, contentRect: target.getBoundingClientRect() } as ResizeObserverEntry], this);
  }
  unobserve(): void {}
  disconnect(): void {}
  takeRecords(): ResizeObserverEntry[] { return []; }
}

globalThis.ResizeObserver = TestResizeObserver;
