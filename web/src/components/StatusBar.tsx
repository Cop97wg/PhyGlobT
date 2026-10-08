import type { BackendHealth } from '../lib/types';

interface Props {
  health: BackendHealth | null;
  backendOk: boolean | null;
}

export default function StatusBar({ health, backendOk }: Props) {
  const state =
    backendOk === null
      ? { cls: 'unknown', text: 'Connecting to backend…' }
      : backendOk
        ? {
            cls: 'ok',
            text: `Backend OK · ${health?.num_scenes ?? 0} scenes · torch ${health?.torch ?? '?'}${
              health?.cuda ? ' · CUDA' : ''
            }`,
          }
        : { cls: 'err', text: 'Backend unreachable — start server/main.py (uvicorn, port 8000)' };

  return (
    <div className={`status ${state.cls}`}>
      <span className="dot" />
      {state.text}
    </div>
  );
}