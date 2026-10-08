import type { AssociationMethod, CheckpointInfo } from '../lib/types';

interface Props {
  checkpoints: CheckpointInfo[];
  selected: string;
  method: AssociationMethod;
  onMethod: (m: AssociationMethod) => void;
  onSelect: (id: string) => void;
  disabled: boolean;
  inferring: boolean;
  error: string | null;
  onInfer: () => void;
  onCompare: () => void;
}

export const METHODS: { id: AssociationMethod; label: string; hint: string }[] = [
  {
    id: 'phyglobt',
    label: 'PhyGlobT (model)',
    hint: 'End-to-end physics-informed model with unbalanced Sinkhorn assignment.',
  },
  {
    id: 'phyglobt_nophysloss',
    label: 'PhyGlobT (no phys-loss)',
    hint: 'Ablation: same architecture, trained without the physics-consistency / focal losses.',
  },
  {
    id: 'lstm',
    label: 'LSTM (deep baseline)',
    hint: 'Learned 2-layer bidirectional LSTM pair encoder (trained on 100% of data).',
  },
  {
    id: 'transformer',
    label: 'Transformer (deep baseline)',
    hint: 'Learned Transformer trajectory pair encoder (replaces the LSTM temporal encoder).',
  },
  {
    id: 'lstm_sinkhorn',
    label: 'LSTM+Sinkhorn (ablation)',
    hint: 'Ablation: PhyGlobT without the physics encoder and without the global topology encoder.',
  },
  { id: 'nn', label: 'Nearest Neighbor (NN)', hint: 'Gated one-to-one nearest-neighbor baseline on kinematics.' },
  { id: 'jpda', label: 'JPDA', hint: 'Joint Probabilistic Data Association baseline (gated marginals).' },
  { id: 'hungarian', label: 'Hungarian', hint: 'Global one-to-one optimal assignment baseline on gated scores.' },
];

const isModel = (m: AssociationMethod) => m === 'phyglobt';

export default function ModelPanel({
  checkpoints,
  selected,
  method,
  onMethod,
  onSelect,
  disabled,
  inferring,
  error,
  onInfer,
  onCompare,
}: Props) {
  const needCkpt = isModel(method);
  return (
    <section className="panel">
      <h2>3. Model</h2>

      <label className="field-label" htmlFor="asm-method">
        Association method
      </label>
      <select
        id="asm-method"
        className="ckpt-select"
        value={method}
        onChange={(e) => onMethod(e.target.value as AssociationMethod)}
      >
        {METHODS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
      <p className="hint">{METHODS.find((m) => m.id === method)?.hint}</p>

      {needCkpt && (
        <>
          <label className="field-label" htmlFor="ckpt">
            Checkpoint
          </label>
          <select
            id="ckpt"
            className="ckpt-select"
            value={selected}
            onChange={(e) => onSelect(e.target.value)}
            disabled={checkpoints.length === 0}
          >
            {checkpoints.length === 0 ? (
              <option value="">(no checkpoints found)</option>
            ) : (
              checkpoints.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id}
                </option>
              ))
            )}
          </select>
        </>
      )}

      <button
        className="primary"
        onClick={onInfer}
        disabled={disabled || inferring || (needCkpt && selected === '')}
      >
        {inferring ? 'Inferring…' : 'Run Association'}
      </button>
      <button className="secondary" onClick={onCompare} disabled={disabled || inferring}>
        Compare all methods
      </button>
      {error && <p className="error">{error}</p>}
      <p className="hint">
        {needCkpt
          ? 'Runs the PhyGlobT model on the selected scene and visualizes the assignment.'
          : 'Runs the selected baseline or ablation (no checkpoint needed) on the selected scene.'}{' '}
        “Compare all methods” runs every method and lists their metrics side by side
        (click a row to show it on the map).
      </p>
    </section>
  );
}