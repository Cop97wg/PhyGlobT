import { useMemo, useRef, useState } from 'react';
import { api } from '../lib/api';
import type { SceneSummary } from '../lib/types';

interface Props {
  scenes: SceneSummary[];
  selected: string | null;
  onSelect: (name: string) => void;
  onUploaded: () => void;
}

interface UploadState {
  file: string;
  ok: boolean;
  message: string;
}

function densityRank(d: string): number {
  if (d.includes('200')) return 2;
  if (d.includes('100')) return 1;
  return 0;
}

export default function DatasetPanel({ scenes, selected, onSelect, onUploaded }: Props) {
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState<'ALL' | 'US' | 'DM'>('ALL');
  const [uploading, setUploading] = useState(false);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    return scenes
      .filter((s) => domain === 'ALL' || s.domain === domain)
      .filter((s) => s.name.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => b.domain.localeCompare(a.domain) || densityRank(b.density) - densityRank(a.density));
  }, [scenes, query, domain]);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    const states: UploadState[] = [];
    for (const f of Array.from(files)) {
      try {
        await api.uploadScene(f);
        states.push({ file: f.name, ok: true, message: 'imported' });
      } catch (e) {
        states.push({ file: f.name, ok: false, message: e instanceof Error ? e.message : String(e) });
      }
    }
    setUploads(states);
    setUploading(false);
    // Reflect imported scenes in the list immediately.
    onUploaded();
    if (fileRef.current) fileRef.current.value = '';
  }

  return (
    <section className="panel">
      <h2>1. Dataset</h2>
      <div className="panel-toolbar">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search scene…"
          aria-label="Search scenes"
        />
        <select
          value={domain}
          onChange={(e) => setDomain(e.target.value as 'ALL' | 'US' | 'DM')}
          aria-label="Filter by domain"
        >
          <option value="ALL">All domains</option>
          <option value="US">US</option>
          <option value="DM">DM (Denmark)</option>
        </select>
      </div>

      <label className="upload-btn" role="button" tabIndex={0} onClick={() => fileRef.current?.click()}>
        {uploading ? 'Importing…' : '⭱ Import local .npy scene(s) — multi-select supported'}
      </label>
      <input
        ref={fileRef}
        type="file"
        accept=".npy"
        multiple
        className="upload-input"
        aria-label="Import local scene files"
        onChange={(e) => void handleFiles(e.target.files)}
      />
      {uploads.length > 0 && (
        <ul className="upload-report">
          {uploads.map((u, i) => (
            <li key={i} className={u.ok ? 'upload-ok' : 'upload-err'}>
              <span className="upload-file">{u.file}</span>
              <span className="upload-msg">{u.message}</span>
            </li>
          ))}
        </ul>
      )}

      {filtered.length === 0 ? (
        <p className="empty">
          No scenes. Start the FastAPI backend, point DATA_DIR at a scenes folder, or import a
          local .npy file above.
        </p>
      ) : (
        <ul className="scene-list">
          {filtered.map((s) => (
            <li key={s.name}>
              <button
                className={selected === s.name ? 'scene-item active' : 'scene-item'}
                onClick={() => onSelect(s.name)}
                title={`${s.num_sensor_a} (A) × ${s.num_sensor_b} (B) tracks, ${s.num_timesteps} steps`}
              >
                <span className="scene-name">{s.name}</span>
                <span className="scene-meta">
                  <span className={`badge badge-${s.domain.toLowerCase()}`}>{s.domain}</span>
                  <span className="badge">{s.density}</span>
                  <span className="badge">
                    {s.num_sensor_a}/{s.num_sensor_b}
                  </span>
                  {s.simulated && <span className="badge badge-sim">SIM</span>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}