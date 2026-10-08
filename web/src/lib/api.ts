import type {
  BackendHealth,
  CheckpointInfo,
  InferenceRequest,
  InferenceProgress,
  SceneDetail,
  SceneSummary,
  SimParams,
  SimulateResponse,
} from './types';

const BASE = '/api';

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${body.slice(0, 300)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => json<BackendHealth>(`${BASE}/health`),

  listScenes: () => json<SceneSummary[]>(`${BASE}/datasets/scenes`),

  listCheckpoints: () => json<CheckpointInfo[]>(`${BASE}/model/checkpoints`),

  getScene: (name: string) =>
    json<SceneDetail>(`${BASE}/datasets/scenes/${encodeURIComponent(name)}`),

  /** Re-simulate a base scene's radar observations with user parameters. */
  simulate: (scene: string, params: Partial<SimParams>) =>
    json<SimulateResponse>(`${BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene, params }),
    }),

  /** Import a local scene .npy file; resolves to the new scene summary. */
  uploadScene: async (file: File): Promise<SceneSummary> => {
    const fd = new FormData();
    fd.append('file', file, file.name);
    const res = await fetch(`${BASE}/datasets/scenes/upload`, { method: 'POST', body: fd });
    if (!res.ok) {
      const msg = await res.json().catch(() => null);
      throw new Error(msg?.detail ?? `HTTP ${res.status}: upload failed`);
    }
    return (await res.json()) as SceneSummary;
  },

  infer: (req: InferenceRequest) =>
    json<{ result: NonNullable<InferenceProgress['message']> extends never ? never : unknown } | never>(
      `${BASE}/model/infer`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      },
    ).then((r) => r as unknown as ReturnType<typeof json<import('./types').AssociationResult>>),
};

/** POST inference with progress polling (streaming not required by backend). */
export async function runInference(
  req: InferenceRequest,
  onProgress: (p: InferenceProgress) => void,
): Promise<import('./types').AssociationResult> {
  onProgress({ status: 'running', stage: 'preparing', message: 'Sending inference request…' });
  const res = await fetch(`${BASE}/model/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    onProgress({ status: 'error', message: `HTTP ${res.status}: ${body.slice(0, 300)}` });
    throw new Error(`Inference failed: ${body.slice(0, 300)}`);
  }
  onProgress({ status: 'running', stage: 'inferring', message: 'Model forward pass…' });
  const data = (await res.json()) as import('./types').AssociationResult;
  onProgress({ status: 'done', message: 'Done.' });
  return data;
}