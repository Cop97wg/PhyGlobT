// Shared API types matching server/main.py response models.

export interface SceneSummary {
  name: string;
  domain: 'US' | 'DM';
  density: string;
  num_sensor_a: number;
  num_sensor_b: number;
  num_timesteps: number;
  duration_min: number;
  has_gt: boolean;
  simulated?: boolean;
}

export interface TrajectoryPoint {
  t: number;
  lat: number;
  lon: number;
  sog: number;
  cog: number;
}

export interface Track {
  id: number;
  sensor: 'A' | 'B';
  mmsi?: string;
  points: TrajectoryPoint[];
}

export interface GroundTruthMatch {
  track_a: number;
  track_b: number;
}

export interface SensorInfo {
  id: number;
  lat: number;
  lon: number;
  max_range_m: number | null;
}

export interface ErrorModel {
  difficulty: string;
  sigma_rho_m: number;
  sigma_theta_deg: number;
  drop_rate: number;
  window_overlap: number;
  source: string;
}

export interface SceneDetail {
  name: string;
  domain: 'US' | 'DM';
  density: string;
  track_note?: string;
  tracks: Track[];
  gt: GroundTruthMatch[] | null;
  sensors?: SensorInfo[];
  error_model?: ErrorModel | null;
  simulated?: boolean;
  base_scene?: string | null;
  sim_params?: SimParams | null;
}

/** Re-simulation parameters (POST /api/simulate) — mirrors server.simulator.SimParams. */
export interface SimParams {
  sigma_rho_m: number;
  sigma_theta_deg: number;
  noise_exponent: number;
  noise_ref_m: number;
  noise_cap: number;
  detection_rate: number;
  sweep_interval_s: number;
  async_jitter_s: number;
  sensor_range_a_m: number | null;
  sensor_range_b_m: number | null;
  seed: number;
}

export interface SimulateResponse {
  name: string;
  summary: SceneSummary;
  info: {
    n_tracks: number;
    tracks_from_truth: number;
    tracks_fallback: number;
    plots: number;
    base_plots: number;
    coordinate_base: string;
  };
}

export interface CheckpointInfo {
  id: string;
  domain: string;
  metrics: Record<string, number> | null;
}

export type AssociationMethod =
  | 'phyglobt'
  | 'phyglobt_nophysloss'
  | 'lstm'
  | 'transformer'
  | 'lstm_sinkhorn'
  | 'nn'
  | 'jpda'
  | 'hungarian';

export interface AssociationResult {
  matches: GroundTruthMatch[];
  soft_matrix: number[][] | null; // soft assignment P (rows = sensor A, cols = sensor B)
  method?: AssociationMethod;
  metrics: {
    overall_f1: number;
    hard_f1: number;
    precision: number;
    recall: number;
    runtime_ms: number;
  };
}

export interface InferenceRequest {
  scene: string;
  checkpoint: string;
  method?: AssociationMethod;
  use_soft_threshold?: boolean;
}

export interface InferenceProgress {
  status: 'running' | 'done' | 'error';
  stage?: string;
  message?: string;
}

export interface BackendHealth {
  status: string;
  version: string;
  torch: string;
  cuda: boolean;
  checkpoints: string[];
  scenes_dir: string;
  num_scenes: number;
}