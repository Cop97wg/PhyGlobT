"""PhyGlobT: Physics-Informed Global Topology Encoding for Multi-Radar
Track-to-Track Association.

Portable subset of the paper implementation: model, preprocessing, dataset and
training entry points. See ``README.md`` for the architecture overview and
reproducibility notes.

Quick start
-----------
.. code-block:: python

    from phyglobt.preprocessing import preprocess_trajectory_unified
    from phyglobt.model import PhysiTrackModel

    proc, ts = preprocess_trajectory_unified(raw_radar_rows, return_original_times=True)
    model = PhysiTrackModel({...})
"""

from __future__ import annotations

from .model import (
    CostPredictor,
    GlobalTopologyEncoder,
    PhysiTrackLoss,
    PhysiTrackModel,
    PhysicalResidualEncoder,
    PositionalEncoding,
    TrajectoryFeatureEncoder,
    TrajectoryTransformerEncoder,
    UnbalancedSinkhornSolver,
    association_loss_balanced,
    focal_loss,
    physical_consistency_loss,
)
from .preprocessing import (
    cartesian_preprocess,
    compute_distance_encoding,
    compute_relative_time_encoding,
    latlon_to_meters,
    preprocess_trajectory_unified,
)
from .dataset import (
    CrossSensorTrajectoryDataset,
    create_data_loaders,
    cross_sensor_collate_fn,
)
from .train import train_one_run

__all__ = [
    # model
    "PhysiTrackModel",
    "PhysicalResidualEncoder",
    "TrajectoryFeatureEncoder",
    "TrajectoryTransformerEncoder",
    "GlobalTopologyEncoder",
    "CostPredictor",
    "UnbalancedSinkhornSolver",
    "PositionalEncoding",
    "focal_loss",
    "association_loss_balanced",
    "physical_consistency_loss",
    "PhysiTrackLoss",
    # preprocessing
    "preprocess_trajectory_unified",
    "cartesian_preprocess",
    "compute_relative_time_encoding",
    "compute_distance_encoding",
    "latlon_to_meters",
    # dataset
    "CrossSensorTrajectoryDataset",
    "create_data_loaders",
    "cross_sensor_collate_fn",
    # training
    "train_one_run",
]

__version__ = "1.0.0"