// lib/api.ts — Video Only Version

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface VideoResult {
  prediction:      "REAL" | "FAKE";
  confidence:      number;
  is_fake:         boolean;
  frames_analyzed: number;
  raw_score:       number;
  filename:        string;
  processing_time: number;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try { const b = await res.json(); message = b?.detail ?? message; } catch {}
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function predictVideo(file: File): Promise<VideoResult> {
  const form = new FormData();
  form.append("file", file);
  return handleResponse<VideoResult>(
    await fetch(`${API_BASE}/predict/video`, { method: "POST", body: form })
  );
}

export async function checkHealth(): Promise<{
  status: string; model: string; device: string;
}> {
  return handleResponse(await fetch(`${API_BASE}/health`));
}