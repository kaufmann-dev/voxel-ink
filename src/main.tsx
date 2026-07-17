import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

async function start(): Promise<void> {
  if (import.meta.env.VITE_WDIO === "true") {
    await import("@wdio/tauri-plugin");
    const { installWdioReferenceFixture } = await import("./testing/wdioReferenceFixture");
    installWdioReferenceFixture();
  }

  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

void start();
