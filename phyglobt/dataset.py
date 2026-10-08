"""Cross-sensor trajectory dataset for PhyGlobT.

Ported from the paper implementation (``Phy_new/two_dataset.py``) with the
``deprecated.preprocessing`` re-export replaced by a direct import from this
package's ``preprocessing`` module.

Each scene ``.npy`` file is a dict with

    observations   : list[np.ndarray]  per-track observations (T, 6) radar rows
    segment_info   : list[dict]        per-track {original_mmsi, sensor_id, ...}
    relation_matrix: np.ndarray        (N, N) cross-sensor ground-truth label matrix
    metadata       : dict              {density_level, coordinate_system, ...}

``sensor_id == 1`` trajectories form set A, ``sensor_id == 2`` form set B.
A crossing pair is positive iff both tracks share the same ``original_mmsi``.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import preprocess_trajectory_unified


class CrossSensorTrajectoryDataset(Dataset):
    """
    Dataset producing the two sensor sets of a scene and their cross-sensor
    association matrix. Returns a 9-tuple:
        (traj_A, traj_B, relations, weights, timestamps_A, timestamps_B,
         mask_A, mask_B, level_id)
    """

    def __init__(self, data_folder, mode="train", max_trajectories=32,
                 time_steps=32, use_interpolation=False,
                 sparsify=True, max_time_gap=1800, max_distance=10.0,
                 mask_augment=False, mask_times=1, mask_trajectory_ratio=0.3,
                 random_seed=42, train_ratio=0.7, test_ratio=0.15,
                 lazy_load=True, max_samples=None):
        self.data_folder = data_folder
        self.mode = mode
        self.max_trajectories = max_trajectories
        self.time_steps = time_steps
        self.use_interpolation = use_interpolation
        self.sparsify = sparsify
        self.max_time_gap = max_time_gap
        self.max_distance = max_distance
        self.mask_augment = mask_augment
        self.mask_times = mask_times
        self.mask_trajectory_ratio = mask_trajectory_ratio
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.lazy_load = lazy_load
        self.max_samples = max_samples

        np.random.seed(random_seed)

        self.all_scenario_files = self._load_scenario_files()
        self.train_files, self.val_files, self.test_files = self._split_dataset()

        if mode == "train":
            self.scenario_files = self.train_files
        elif mode == "val":
            self.scenario_files = self.val_files
        elif mode == "test":
            self.scenario_files = self.test_files
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if self.lazy_load:
            self.scenarios = None
        else:
            self.scenarios = self._load_scenarios()

        if self.mask_augment and self.mode == "train":
            self.original_length = len(self.scenario_files)
            self.total_length = self.original_length * self.mask_times
        else:
            self.original_length = len(self.scenario_files)
            self.total_length = self.original_length

    def _load_scenario_files(self):
        files = []
        train_pattern = os.path.join(self.data_folder, "train", "**", "*.npy")
        files = glob.glob(train_pattern, recursive=True)
        if len(files) == 0:
            mode_folder = os.path.join(self.data_folder, self.mode)
            if os.path.exists(mode_folder):
                files = glob.glob(os.path.join(mode_folder, "*.npy"))
        if len(files) == 0:
            files = glob.glob(os.path.join(self.data_folder, "**", "*.npy"), recursive=True)
        if len(files) == 0:
            files = glob.glob(os.path.join(self.data_folder, "*.npy"))
        return files

    def _split_dataset(self):
        all_files = self.all_scenario_files.copy()
        if len(all_files) == 0:
            return [], [], []
        np.random.shuffle(all_files)
        if self.max_samples is not None and self.max_samples > 0:
            all_files = all_files[: self.max_samples]
        n_total = len(all_files)
        n_train = int(n_total * self.train_ratio)
        n_test = int(n_total * self.test_ratio)
        n_val = n_total - n_train - n_test
        if n_val == 0 and n_train > 0:
            n_train -= 1
            n_val = 1
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        return train_files, val_files, test_files

    def _load_scenarios(self):
        scenarios = []
        for fp in self.scenario_files:
            try:
                data = np.load(fp, allow_pickle=True).item()
                scenarios.append(data)
            except Exception as e:
                print(f"Error loading {fp}: {e}")
        return scenarios

    def _preprocess_trajectory(self, trajectory, return_timestamps=False):
        if return_timestamps:
            processed_traj, timestamps = preprocess_trajectory_unified(
                trajectory, self.time_steps, self.use_interpolation, return_original_times=True
            )
            return processed_traj, timestamps
        return preprocess_trajectory_unified(
            trajectory, self.time_steps, self.use_interpolation
        )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.mask_augment and self.mode == "train":
            original_idx = idx % self.original_length
            mask_id = idx // self.original_length
        else:
            original_idx = idx
            mask_id = 0

        if self.lazy_load:
            file_path = self.scenario_files[original_idx]
            scenario = np.load(file_path, allow_pickle=True).item()
        else:
            scenario = self.scenarios[original_idx]

        observations = scenario.get("observations", [])
        segment_info = scenario.get("segment_info", [])

        sensor1_trajs, sensor2_trajs = [], []
        sensor1_info, sensor2_info = [], []

        for i, traj in enumerate(observations):
            if len(traj) < 10:
                continue
            sensor_id = segment_info[i].get("sensor_id", -1)
            if sensor_id == 1:
                if self.sparsify:
                    proc, tstamp = self._preprocess_trajectory(traj, return_timestamps=True)
                    sensor1_trajs.append((proc, tstamp))
                else:
                    proc = self._preprocess_trajectory(traj)
                    sensor1_trajs.append((proc, None))
                sensor1_info.append(segment_info[i])
            elif sensor_id == 2:
                if self.sparsify:
                    proc, tstamp = self._preprocess_trajectory(traj, return_timestamps=True)
                    sensor2_trajs.append((proc, tstamp))
                else:
                    proc = self._preprocess_trajectory(traj)
                    sensor2_trajs.append((proc, None))
                sensor2_info.append(segment_info[i])

        if len(sensor1_trajs) == 0 or len(sensor2_trajs) == 0:
            return self._get_fallback_batch()

        if len(sensor1_trajs) > self.max_trajectories:
            indices = np.random.choice(len(sensor1_trajs), self.max_trajectories, replace=False)
            sensor1_trajs = [sensor1_trajs[i] for i in indices]
            sensor1_info = [sensor1_info[i] for i in indices]
        if len(sensor2_trajs) > self.max_trajectories:
            indices = np.random.choice(len(sensor2_trajs), self.max_trajectories, replace=False)
            sensor2_trajs = [sensor2_trajs[i] for i in indices]
            sensor2_info = [sensor2_info[i] for i in indices]

        n1 = len(sensor1_trajs)
        n2 = len(sensor2_trajs)
        traj1_array = np.stack([t[0] for t in sensor1_trajs])
        traj2_array = np.stack([t[0] for t in sensor2_trajs])
        if self.sparsify:
            ts1_array = np.stack([t[1] for t in sensor1_trajs])
            ts2_array = np.stack([t[1] for t in sensor2_trajs])
        else:
            ts1_array = np.zeros((n1, self.time_steps))
            ts2_array = np.zeros((n2, self.time_steps))

        # Ground-truth association matrix from shared MMSI
        relation_mat = np.zeros((n1, n2))
        mmsi_to_indices1 = {}
        mmsi_to_indices2 = {}
        for idx1, info in enumerate(sensor1_info):
            mmsi = info.get("original_mmsi", "")
            mmsi_to_indices1.setdefault(mmsi, []).append(idx1)
        for idx2, info in enumerate(sensor2_info):
            mmsi = info.get("original_mmsi", "")
            mmsi_to_indices2.setdefault(mmsi, []).append(idx2)
        for mmsi, indices1 in mmsi_to_indices1.items():
            indices2 = mmsi_to_indices2.get(mmsi, [])
            for i in indices1:
                for j in indices2:
                    relation_mat[i, j] = 1.0

        num_pos = relation_mat.sum()
        num_neg = relation_mat.size - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        weight_mat = np.where(relation_mat == 1, pos_weight, 1.0)

        pad_n1 = self.max_trajectories - n1
        pad_n2 = self.max_trajectories - n2
        if pad_n1 > 0:
            pad_traj1 = np.zeros((pad_n1, self.time_steps, 5))
            traj1_array = np.concatenate([traj1_array, pad_traj1], axis=0)
            pad_ts1 = np.zeros((pad_n1, self.time_steps))
            ts1_array = np.concatenate([ts1_array, pad_ts1], axis=0)
        if pad_n2 > 0:
            pad_traj2 = np.zeros((pad_n2, self.time_steps, 5))
            traj2_array = np.concatenate([traj2_array, pad_traj2], axis=0)
            pad_ts2 = np.zeros((pad_n2, self.time_steps))
            ts2_array = np.concatenate([ts2_array, pad_ts2], axis=0)

        full_relation = np.zeros((self.max_trajectories, self.max_trajectories))
        full_relation[:n1, :n2] = relation_mat
        full_weight = np.ones((self.max_trajectories, self.max_trajectories))
        full_weight[:n1, :n2] = weight_mat

        mask1 = np.zeros(self.max_trajectories)
        mask1[:n1] = 1
        mask2 = np.zeros(self.max_trajectories)
        mask2[:n2] = 1

        if self.mask_augment and self.mode == "train" and mask_id > 0:
            traj1_array, mask1 = self._apply_random_mask(traj1_array, mask1, mask_id)
            traj2_array, mask2 = self._apply_random_mask(traj2_array, mask2, mask_id + 1000)
            for i in range(self.max_trajectories):
                if mask1[i] == 0:
                    full_relation[i, :] = 0
                    full_weight[i, :] = 0
                if mask2[i] == 0:
                    full_relation[:, i] = 0
                    full_weight[:, i] = 0

        density_level = scenario.get("metadata", {}).get("density_level", "unknown")
        level_map = {"50_100": 0, "100_200": 1, "200_plus": 2, "unknown": -1}
        level_id = level_map.get(density_level, -1)

        return (torch.FloatTensor(traj1_array),
                torch.FloatTensor(traj2_array),
                torch.FloatTensor(full_relation),
                torch.FloatTensor(full_weight),
                torch.FloatTensor(ts1_array),
                torch.FloatTensor(ts2_array),
                torch.FloatTensor(mask1),
                torch.FloatTensor(mask2),
                torch.tensor(level_id, dtype=torch.long))

    def _apply_random_mask(self, trajectories, mask_indicators, mask_id=0):
        num_traj = len(trajectories)
        min_keep = max(1, int(num_traj * (1 - self.mask_trajectory_ratio)))
        max_keep = num_traj - 1
        np.random.seed(self.random_seed + mask_id)
        keep_count = np.random.randint(min_keep, max_keep + 1)
        keep_indices = np.sort(np.random.choice(num_traj, keep_count, replace=False))
        new_mask = np.zeros(num_traj)
        new_mask[keep_indices] = 1
        masked_trajs = trajectories.copy()
        for i in range(num_traj):
            if new_mask[i] == 0:
                masked_trajs[i] = 0
        np.random.seed(self.random_seed)
        return masked_trajs, new_mask

    def _get_fallback_batch(self):
        dummy_traj = torch.zeros((self.max_trajectories, self.time_steps, 5))
        dummy_ts = torch.zeros((self.max_trajectories, self.time_steps))
        dummy_mask = torch.zeros(self.max_trajectories)
        dummy_rel = torch.zeros((self.max_trajectories, self.max_trajectories))
        dummy_weight = torch.ones((self.max_trajectories, self.max_trajectories))
        return (dummy_traj, dummy_traj, dummy_rel, dummy_weight,
                dummy_ts, dummy_ts, dummy_mask, dummy_mask,
                torch.tensor(-1, dtype=torch.long))


def cross_sensor_collate_fn(batch):
    transposed = list(zip(*batch))
    collated = []
    for i, items in enumerate(transposed):
        if i == 8:
            collated.append(torch.tensor(items, dtype=torch.long))
        else:
            if all(isinstance(item, torch.Tensor) for item in items):
                collated.append(torch.stack(items))
            else:
                tensor_items = [item if isinstance(item, torch.Tensor) else torch.tensor(item) for item in items]
                collated.append(torch.stack(tensor_items))
    return tuple(collated)


def create_data_loaders(data_folder, config):
    """
    Build train/val/test DataLoaders and datasets.

    Each batch is a 9-tuple:
    traj_A, traj_B, relations, weights, timestamps_A, timestamps_B,
    mask_A, mask_B, level_ids
    """
    train_dataset = CrossSensorTrajectoryDataset(
        data_folder=data_folder, mode="train",
        max_trajectories=config.get("max_trajectories", 32),
        time_steps=config.get("time_steps", 32),
        use_interpolation=config.get("use_interpolation", False),
        sparsify=config.get("sparsify", True),
        max_time_gap=config.get("max_time_gap", 1800),
        max_distance=config.get("max_distance", 10.0),
        mask_augment=config.get("mask_augment", False),
        mask_times=config.get("mask_times", 1),
        mask_trajectory_ratio=config.get("mask_trajectory_ratio", 0.3),
        random_seed=config.get("random_seed", 42),
        train_ratio=config.get("train_ratio", 0.7),
        test_ratio=config.get("test_ratio", 0.15),
    )
    val_dataset = CrossSensorTrajectoryDataset(
        data_folder=data_folder, mode="val",
        max_trajectories=config.get("max_trajectories", 32),
        time_steps=config.get("time_steps", 32),
        use_interpolation=config.get("use_interpolation", False),
        sparsify=config.get("sparsify", True),
        max_time_gap=config.get("max_time_gap", 1800),
        max_distance=config.get("max_distance", 10.0),
        mask_augment=False,
        random_seed=config.get("random_seed", 42),
        train_ratio=config.get("train_ratio", 0.7),
        test_ratio=config.get("test_ratio", 0.15),
    )
    test_dataset = None
    if config.get("test_ratio", 0.15) > 0:
        test_dataset = CrossSensorTrajectoryDataset(
            data_folder=data_folder, mode="test",
            max_trajectories=config.get("max_trajectories", 32),
            time_steps=config.get("time_steps", 32),
            use_interpolation=config.get("use_interpolation", False),
            sparsify=config.get("sparsify", True),
            max_time_gap=config.get("max_time_gap", 1800),
            max_distance=config.get("max_distance", 10.0),
            mask_augment=False,
            random_seed=config.get("random_seed", 42),
            train_ratio=config.get("train_ratio", 0.7),
            test_ratio=config.get("test_ratio", 0.15),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size", 8), shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True), collate_fn=cross_sensor_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size", 8), shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True), collate_fn=cross_sensor_collate_fn,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=config.get("batch_size", 8), shuffle=False,
            num_workers=config.get("num_workers", 2),
            pin_memory=config.get("pin_memory", True), collate_fn=cross_sensor_collate_fn,
        )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset