import { useState } from 'react';
import { api } from '../lib/api';
import type { SceneDetail, SimParams } from '../lib/types';

interface Props {
  scene: SceneDetail | null;
  /** Called with the new simulated scene name after a successful run. */
  onSimulated: (name: string) => void;
}

/** Defaults mirror the original generator (radar_sim.json). */
const DEFAULTS: SimParams = {
  sigma_rho_m: 30,
  sigma_theta_deg: 0.5,
  noise_exponent: 0.8,
  noise_ref_m: 10000,
  noise_cap: 4,
  detection_rate: 0.68,
  sweep_interval_s: 10,
  async_jitter_s: 2,
  sensor_range_a_m: null,
  sensor_range_b_m: null,
  seed: 0,
};

const PRESETS: { id: string; label: string; patch: Partial<SimParams> }[] = [
  {
    id: 'generator',
    label: 'Generator default (σρ 30 m · det 68% · 10 s sweep)',
    patch: { sigma_rho_m: 30, sigma_theta_deg: 0.5, noise_exponent: 0.8, detection_rate: 0.68, sweep_interval_s: 10, async_jitter_s: 2 },
  },
  {
    id: 'clean',
    label: 'Clean (no noise · no misses)',
    patch: { sigma_rho_m: 0, sigma_theta_deg: 0, detection_rate: 1, async_jitter_s: 0 },
  },
  {
    id: 'noisy',
    label: 'High noise (σρ 60 m · det 50% · ±3 s)',
    patch: { sigma_rho_m: 60, sigma_theta_deg: 1.0, detection_rate: 0.5, sweep_interval_s: 12, async_jitter_s: 3 },
  },
  {
    id: 'dense',
    label: 'Fast scan (12 rpm · det 85%)',
    patch: { sigma_rho_m: 30, sigma_theta_deg: 0.5, detection_rate: 0.85, sweep_interval_s: 5, async_jitter_s: 1 },
  },
];

export default function SimPanel({ scene, onSimulated }: Props) {
  const [p, setP] = useState<SimParams>(DEFAULTS);
  const [preset, setPreset] = useState('generator');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const up = (k: keyof SimParams, v: number | null) => setP((s) => ({ ...s, [k]: v }));

  const numField = (
    label: string,
    key: keyof SimParams,
    step: number,
    hint?: string,
  ) => (
    <label className="sim-field" title={hint}>
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={p[key] as number}
        onChange={(e) => up(key, e.target.value === '' ? 0 : Number(e.target.value))}
      />
    </label>
  );

  const rangeField = (label: string, key: 'sensor_range_a_m' | 'sensor_range_b_m', defKm: number | null) => (
    <label className="sim-field" title="Empty = keep the base scene's value">
      <span>{label}</span>
      <input
        type="number"
        step={1000}
        placeholder={defKm ? `${Math.round(defKm / 1000)} km` : '—'}
        value={p[key] ?? ''}
        onChange={(e) => up(key, e.target.value === '' ? null : Number(e.target.value))}
      />
    </label>
  );

  const rangeA = scene?.sensors?.find((s) => s.id === 1)?.max_range_m ?? null;
  const rangeB = scene?.sensors?.find((s) => s.id === 2)?.max_range_m ?? null;

  async function run() {
    if (!scene || scene.simulated) return;
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const r = await api.simulate(scene.name, p);
      setMsg(
        `Created ${r.name} — ${r.info.plots} plots (base: ${r.info.base_plots}), ` +
          `${r.info.tracks_from_truth} tracks re-simulated from AIS truth.`,
      );
      onSimulated(r.name);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const disabled = !scene || busy;

  return (
    <section className="panel">
      <h2>2. Radar Simulation</h2>
      <p className="hint">
        Re-simulate both radars from the base scene&apos;s AIS ground truth: sweep-grid sampling,
        range/azimuth noise ∝ (r/10 km)<sup>0.8</sup>, random misses, sensor asynchrony. Track ids
        and ground-truth pairs are preserved, so any model can be evaluated on the result. You can
        re-run as many times as you like (different seeds / parameters produce new scenes).
      </p>

      <label className="field-label" htmlFor="sim-preset">
        Preset
      </label>
      <select
        id="sim-preset"
        className="ckpt-select"
        value={preset}
        onChange={(e) => {
          const id = e.target.value;
          setPreset(id);
          const pr = PRESETS.find((x) => x.id === id);
          if (pr) setP((s) => ({ ...s, ...pr.patch }));
        }}
      >
        {PRESETS.map((x) => (
          <option key={x.id} value={x.id}>
            {x.label}
          </option>
        ))}
      </select>

      <div className="sim-grid">
        {numField('σρ (m @10 km)', 'sigma_rho_m', 5, 'Range-noise level at the reference range')}
        {numField('σθ (° @10 km)', 'sigma_theta_deg', 0.05, 'Azimuth-noise level at the reference range')}
        {numField('Detection rate', 'detection_rate', 0.01, 'P(a sweep yields a plot); 0.68 ≙ Pd 0.85 × (1−drop 0.2)')}
        {numField('Sweep interval (s)', 'sweep_interval_s', 1, 'Antenna period: 10 s ≙ 6 rpm, 5 s ≙ 12 rpm')}
        {numField('Async jitter (±s)', 'async_jitter_s', 0.5, 'Random sweep-offset between the two radars')}
        {numField('Seed (0 = random)', 'seed', 1, 'Fix a seed for reproducible simulations')}
        {rangeField('Sensor A range', 'sensor_range_a_m', rangeA)}
        {rangeField('Sensor B range', 'sensor_range_b_m', rangeB)}
      </div>

      <button className="primary" disabled={disabled} onClick={() => void run()}>
        {busy ? 'Simulating…' : '▶ Run simulation'}
      </button>
      {!scene && <p className="hint">Load a scene first.</p>}
      {msg && <p className="upload-ok sim-report">{msg}</p>}
      {err && <p className="error">{err}</p>}
    </section>
  );
}
