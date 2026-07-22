import { useState } from "react";

type ControlEndpoint = "/start-run" | "/stop-run" | "/dev-reload";

interface ControlResult {
  ok: boolean;
  message: string;
}

async function postControl(endpoint: ControlEndpoint): Promise<ControlResult> {
  try {
    const response = await fetch(endpoint, { method: "POST" });
    const body = await response.json();

    if (!response.ok) {
      return { ok: false, message: body.error ?? `Request failed (${response.status})` };
    }

    return { ok: true, message: JSON.stringify(body) };
  } catch (err) {
    return { ok: false, message: err instanceof Error ? err.message : "Unknown error" };
  }
}

export default function App() {
  const [status, setStatus] = useState<string>("Idle");
  const [busy, setBusy] = useState<ControlEndpoint | null>(null);
  // Cache-busts the <img> src on demand so the browser reconnects the
  // multipart/x-mixed-replace stream after a start/stop/reload cycle.
  const [streamKey, setStreamKey] = useState(0);

  const runControl = async (endpoint: ControlEndpoint, label: string) => {
    setBusy(endpoint);
    setStatus(`${label}...`);

    const result = await postControl(endpoint);

    setStatus(result.ok ? `${label}: ${result.message}` : `${label} failed: ${result.message}`);
    setStreamKey((key) => key + 1);
    setBusy(null);
  };

  return (
    <div className="app">
      <header>
        <h1>SAM 3.1 Live Inference Worker</h1>
        <p className="status">{status}</p>
      </header>

      <div className="stream-frame">
        <img
          key={streamKey}
          src="/video-stream"
          alt="Live SAM 3.1 segmented video stream"
          className="stream"
        />
      </div>

      <div className="controls">
        <button
          disabled={busy !== null}
          onClick={() => runControl("/start-run", "Kick Off SAM 3")}
        >
          Kick Off SAM 3
        </button>
        <button
          disabled={busy !== null}
          onClick={() => runControl("/stop-run", "Pause Inference")}
        >
          Pause Inference
        </button>
        <button
          disabled={busy !== null}
          onClick={() => runControl("/dev-reload", "Hot-Reload Python Code")}
        >
          Hot-Reload Python Code
        </button>
      </div>
    </div>
  );
}
