import type { AssociationResult, SceneDetail } from '../lib/types';

interface Props {
  scene: SceneDetail | null;
  result: AssociationResult | null;
  inferring: boolean;
  compare: AssociationResult[];
  onPickCompare: (r: AssociationResult) => void;
}

const METHOD_LABEL: Record<string, string> = {
  phyglobt: 'PhyGlobT',
  phyglobt_nophysloss: 'PhyGlobT (no phys-loss)',
  lstm: 'LSTM',
  transformer: 'Transformer',
  lstm_sinkhorn: 'LSTM+Sinkhorn',
  nn: 'NN',
  jpda: 'JPDA',
  hungarian: 'Hungarian',
};

function MetricCard({ label, value, suffix, hint }: { label: string; value: string; suffix?: string; hint?: string }) {
  return (
    <div className="metric" title={hint}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">
        {value}
        {suffix && <span className="metric-suffix">{suffix}</span>}
      </span>
    </div>
  );
}

export default function ResultPanel({ scene, result, inferring, compare, onPickCompare }: Props) {
  if (!scene) {
    return (
      <section className="panel">
        <h2>4. Results</h2>
        <p className="empty">Load a scene to see metrics.</p>
      </section>
    );
  }

  const nGt = scene.gt?.length ?? 0;
  const nPred = result?.matches.length ?? 0;
  // True positives: predicted pairs that actually match a ground-truth pair.
  const nRecovered =
    result && scene.gt
      ? result.matches.filter((m) =>
          scene.gt!.some((g) => g.track_a === m.track_a && g.track_b === m.track_b),
        ).length
      : 0;
  const bestF1 = compare.reduce((mx, c) => Math.max(mx, c.metrics.overall_f1), 0);

  return (
    <section className="panel result-panel">
      <h2>4. Results</h2>

      <div className="scene-stats">
        <MetricCard label="Tracks (A/B)" value={`${scene.tracks.filter((t) => t.sensor === 'A').length}/${scene.tracks.filter((t) => t.sensor === 'B').length}`} />
        <MetricCard label="Ground truth" value={String(nGt)} />
        {result && <MetricCard label="Predicted" value={String(nPred)} />}
      </div>

      {inferring && <p className="hint">Inference running…</p>}

      {result && (
        <>
          <h3>Metrics{result.method ? ` — ${METHOD_LABEL[result.method] ?? result.method}` : ''}</h3>
          <div className="scene-stats">
            <MetricCard label="Overall F1" value={(result.metrics.overall_f1 * 100).toFixed(1)} suffix="%" hint="F1 over all predicted vs ground-truth pairs (precision & recall averaged)." />
            <MetricCard label="Hard F1" value={(result.metrics.hard_f1 * 100).toFixed(1)} suffix="%" hint="Per-scene demo: hard F1 equals overall F1 (scenes are evaluated individually)." />
            <MetricCard label="Precision" value={(result.metrics.precision * 100).toFixed(1)} suffix="%" hint="TP / (TP + FP): share of predicted pairs that are correct." />
            <MetricCard label="Recall" value={(result.metrics.recall * 100).toFixed(1)} suffix="%" hint="TP / (TP + FN): share of ground-truth pairs recovered." />
            <MetricCard label="Runtime" value={result.metrics.runtime_ms.toFixed(0)} suffix="ms" />
          </div>
        </>
      )}

      <div className="matrix-block">
        <h3>Matching attribution</h3>
        <p className="hint">
          {result
            ? `${nRecovered} of ${nGt} ground-truth pairs recovered${result.soft_matrix ? ' (soft Sinkhorn plan available in API)' : ''}.`
            : 'Run association to see predictions.'}
        </p>
      </div>

      {compare.length > 0 && (
        <div className="matrix-block">
          <h3>Method comparison</h3>
          <table className="cmp-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>#Pred</th>
                <th>ms</th>
              </tr>
            </thead>
            <tbody>
              {compare.map((c, i) => {
                const label = METHOD_LABEL[c.method ?? ''] ?? c.method ?? `method-${i}`;
                return (
                  <tr
                    key={`${c.method ?? i}`}
                    className={`${result?.method === c.method ? 'active' : ''} ${c.metrics.overall_f1 >= bestF1 ? 'best' : ''}`}
                    onClick={() => onPickCompare(c)}
                    title="Show this method's result on the map"
                  >
                    <td>{label}</td>
                    <td>{(c.metrics.overall_f1 * 100).toFixed(1)}%</td>
                    <td>{(c.metrics.precision * 100).toFixed(1)}%</td>
                    <td>{(c.metrics.recall * 100).toFixed(1)}%</td>
                    <td>{c.matches.length}</td>
                    <td>{c.metrics.runtime_ms.toFixed(0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="hint">Click a row to show that method on the map. ★ marks the best F1.</p>
        </div>
      )}
    </section>
  );
}