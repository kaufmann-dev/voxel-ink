import type { Capabilities, Options } from "@wdio/types";

const binary = process.platform === "win32"
  ? "./src-tauri/target/debug/voxel-ink.exe"
  : "./src-tauri/target/debug/voxel-ink";

type TauriCapabilities = Capabilities.RequestedStandaloneCapabilities & {
  "tauri:options": { application: string };
};

export const config: Options.Testrunner & { capabilities: TauriCapabilities[] } = {
  runner: "local",
  specs: ["./test/smoke/**/*.spec.ts"],
  maxInstances: 1,
  services: [["@wdio/tauri-service", {
    appBinaryPath: binary,
    driverProvider: "embedded",
    embeddedPort: 4445,
    captureBackendLogs: true,
    captureFrontendLogs: true,
  }]],
  capabilities: [{
    browserName: "tauri",
    "tauri:options": { application: binary },
  }],
  logLevel: "warn",
  bail: 1,
  waitforTimeout: 10_000,
  connectionRetryTimeout: 90_000,
  connectionRetryCount: 1,
  framework: "mocha",
  reporters: ["spec"],
  mochaOpts: { ui: "bdd", timeout: 60_000 },
};
