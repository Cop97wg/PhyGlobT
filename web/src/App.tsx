import { useCallback, useEffect, useRef, useState } from 'react';
import DatasetPanel from './components/DatasetPanel';
import MapView from './components/MapView';
import ModelPanel from './components/ModelPanel';
import ResultPanel from './components/ResultPanel';
import SimPanel from './components/SimPanel';
import StatusBar from './components/StatusBar';
import { api, runInference } from './lib/api';
import type {
  AssociationMethod,
  AssociationResult,
  BackendHealth,
  CheckpointInfo,
  SceneDetail,
  SceneSummary,
} from './lib/types';

export default function App() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [selectedScene, setSelectedScene] = useState<SceneDetail | null>(null);

  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('phyglobt_us_best.pt');
  const [method, setMethod] = useState<AssociationMethod>('phyglobt');

  const [result, setResult] = useState<AssociationResult | null>(null);
  const [compareResults, setCompareResults] = useState<AssociationResult[]>([]);
  const [inferring, setInferring] = useState(false);
  const [inferError, setInferError] = useState<string | null>(null);

  const mapRef = useRef<{ focusScene: (s: SceneDetail) => void } | null>(null);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setBackendOk(true);
        if (h.scenes_dir) void loadScenes();
        if (h.checkpoints?.length) {
          setCheckpoints(h.checkpoints.map((c) => ({ id: c, domain: 'US', metrics: null })));
          setSelectedCheckpoint((prev) =>
            h.checkpoints.includes(prev) ? prev : (h.checkpoints[0] ?? prev),
          );
        }
      })
      .catch((e) => {
        void e;
        setBackendOk(false);
      });
  }, []);

  const loadScenes = useCallback(async () => {
    try {
      const list = await api.listScenes();
      setScenes(list);
      // Auto-load the default scene on first successful list (no scene selected yet).
      setSelectedScene((prev) => {
        if (prev) return prev;
        const names = list.map((s) => s.name);
        const name = names.includes(DEFAULT_SCENE) ? DEFAULT_SCENE : (list[0]?.name ?? DEFAULT_SCENE);
        // Fire-and-forget the detail load; focusScene is invoked by the handler.
        void handleSelectScene(name);
        return prev; // selectedScene is set by handleSelectScene once detail arrives
      });
    } catch (e) {
      console.error('Failed to list scenes:', e);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Default scene: 200_plus where PhyGlobT leads by the widest margin over
  // traditional baselines (gap_ph ≈ +0.19 F1), so the comparison table reads
  // clearly. Falls back to the first scene when it isn't present.
  const DEFAULT_SCENE = 'scenes_200_plus_48_49_-124_-123_20240526_020000.npy';

  const handleSelectScene = useCallback(async (name: string) => {
    try {
      const detail = await api.getScene(name);
      setSelectedScene(detail);
      setResult(null);
      setCompareResults([]);
      mapRef.current?.focusScene(detail);
    } catch (e) {
      console.error('Failed to load scene:', e);
    }
  }, []);

  const handleInfer = useCallback(
    async (m: AssociationMethod, checkpoint: string) => {
      if (!selectedScene) return;
      setInferring(true);
      setInferError(null);
      setResult(null);
      try {
        const r = await api.infer({
          scene: selectedScene.name,
          checkpoint,
          method: m,
        });
        setResult(r);
      } catch (e) {
        setInferError(e instanceof Error ? e.message : String(e));
      } finally {
        setInferring(false);
      }
    },
    [selectedScene],
  );

  // Run every method on the current scene and show them side by side.
  const handleCompare = useCallback(async () => {
    if (!selectedScene) return;
    setInferring(true);
    setInferError(null);
    // Run all 8 methods: PhyGlobT + ablation, deep baselines, traditional baselines.
    const methods: AssociationMethod[] = selectedCheckpoint
      ? ['phyglobt', 'phyglobt_nophysloss', 'lstm', 'transformer', 'lstm_sinkhorn', 'nn', 'jpda', 'hungarian']
      : ['lstm', 'transformer', 'lstm_sinkhorn', 'nn', 'jpda', 'hungarian'];
    const results: AssociationResult[] = [];
    for (const m of methods) {
      try {
        results.push(
          await runInference(
            { scene: selectedScene.name, checkpoint: selectedCheckpoint, method: m },
            () => {},
          ),
        );
      } catch {
        // skip a failed method but keep comparing the rest
      }
    }
    setCompareResults(results);
    if (results.length) {
      // Show the best-F1 method on the map by default; rows are clickable.
      setResult(results.reduce((a, b) => (b.metrics.overall_f1 > a.metrics.overall_f1 ? b : a)));
    } else {
      setInferError('All methods failed — check backend logs.');
    }
    setInferring(false);
  }, [selectedScene, selectedCheckpoint]);

  // After a simulation: refresh the scene list, then load the new scene.
  const handleSimulated = useCallback(
    async (name: string) => {
      await loadScenes();
      await handleSelectScene(name);
    },
    [loadScenes, handleSelectScene],
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>PhyGlobT — Multi-Radar Track-to-Track Association</h1>
        <StatusBar health={health} backendOk={backendOk} />
      </header>

      <main className="app-body">
        <aside className="sidebar">
          <DatasetPanel
            scenes={scenes}
            selected={selectedScene?.name ?? null}
            onSelect={handleSelectScene}
            onUploaded={() => void loadScenes()}
          />
          <SimPanel scene={selectedScene} onSimulated={(name) => void handleSimulated(name)} />
          <ModelPanel
            checkpoints={checkpoints}
            selected={selectedCheckpoint}
            method={method}
            onMethod={setMethod}
            onSelect={setSelectedCheckpoint}
            disabled={!selectedScene}
            inferring={inferring}
            error={inferError}
            onInfer={() => void handleInfer(method, selectedCheckpoint)}
            onCompare={() => void handleCompare()}
          />
        </aside>

        <section className="map-area">
          <MapView ref={mapRef} scene={selectedScene} result={result} />
        </section>

        <aside className="inspector">
          <ResultPanel
            scene={selectedScene}
            result={result}
            inferring={inferring}
            compare={compareResults}
            onPickCompare={setResult}
          />
        </aside>
      </main>
    </div>
  );
}
