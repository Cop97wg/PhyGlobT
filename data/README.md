# Data Sources

PhyGlobT is trained and evaluated on **real AIS vessel trajectories** from two open
data portals. Raw AIS messages are converted to radar-like observations through the
IMO noise model `MSC.192(79)` (radial/azimuthal range-dependent errors). The resulting
scene files (`.npy`) are not redistributed here due to size (~GBs), but can be
regenerated with the scripts in `../phyglobt/preprocessing.py` and `../phyglobt/dataset.py`.

| Domain | Region | Source | Download link |
|---|---|---|---|
| US | Gulf of Mexico / US coastal waters | NOAA MarineCadastre Vessel Traffic (AIS) | <https://marinecadastre.gov/ais/> |
| DM (Denmark) | Baltic Sea & North Sea (Danish waters) | Danish Maritime Authority (DMA) historical AIS | <https://www.dma.dk/safety-at-sea/navigational-information/ais-data> (bulk files: <https://web.ais.dk/aisdata/>) |

---

## 1. US — NOAA MarineCadastre AIS

- **Portal:** <https://marinecadastre.gov/ais/>
- **Dataset name:** Vessel Traffic (AIS) — enabled by the US Coast Guard and Bureau of
  Ocean Energy Management (BOEM).
- **Contents:** nationwide broadcast AIS messages (position reports + voyage data),
  available as per-year **CSV or File Geodatabase** downloads, post 2015 in monthly
  zips + 2024+ in dynamic zips.
- **Fields used by PhyGlobT:** `MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`,
  `Heading`, `VesselType`.
- **Processing:** keep merchant/passenger/fishing class A messages → resample to
  uniform time steps → keep periods with ≥ `min_length` consecutive reports.

```bash
# Example (pulling the 2024 Gulf of Mexico monthly zip via the portal's ZIP URL pattern)
# 1) Download from https://marinecadastre.gov/ais/ (select year 2024 → zone → zip)
# 2) Unzip and load CSV columns: MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselType
```

---

## 2. DM — Danish Maritime Authority (DMA) Historical AIS

- **Portal:** <https://www.dma.dk/safety-at-sea/navigational-information/ais-data>
- **Bulk file index:** <https://web.ais.dk/aisdata/> (files `aisdk-YYYY-MM-DD.zip`,
  from March 2006 to present; 2014+ available as daily/weekly/monthly zips).
- **Contents:** shore-based AIS messages collected at Danish coastal stations
  (Baltic Sea + North Sea area ~lat 54–59°N, lon 3–17°E).
- **Fields used by PhyGlobT:** `Timestamp`, `MMSI`, `Latitude`, `Longitude`, `SOG`,
  `COG`, `Heading`, `NavStatus`, `ShipType`.
- **Format note:** DMA provides raw AIS CSV columns (Type1/2/3, Type5, etc. filtered
  in `preprocessing.py`).

```bash
# Example: download a single day directly
Invoke-WebRequest "https://web.ais.dk/aisdata/aisdk-2024-09-15.zip" -OutFile "aisdk-2024-09-15.zip"
# Unzip → CSV(s) with columns: Timestamp,Type,MMSI,Latitude,Longitude,SOG,COG,Heading,...
```

---

## 3. Scene file format (.npy)

Each scene is a Python `np.load(..., allow_pickle=True)` dictionary (see
`../phyglobt/dataset.py`):

```python
scene = np.load("scene_000231.npy", allow_pickle=True).item()
scene.keys()
# dict_keys(['observations', 'segment_info', 'metadata'])

scene["observations"]        # radar observations, list of np.ndarray [T, 5] or [T, 6]
                             # [lat, lon, sog, cog, heading, timestamp] (6-col radar format)
scene["segment_info"]        # list of dicts: {segment_id, sensor_id, start_idx, ...}
scene["metadata"]            # dict: {domain, density, num_trajectories, duration, ...}
```

Observations follow the trajectory format `[time, lat, lon, sog, cog]` (5 columns) or
radar format `[lat, lon, sog, cog, heading, timestamp]` (6 columns). See
`../phyglobt/preprocessing.py` for the exact column mapping.

---

## 4. Regenerating a scene dataset

1. Download raw AIS (US and/or DM) as above.
2. Run the preprocessing pipeline to convert trajectories → radar observations
   (IMO noise model) → scene `.npy` files:

```bash
python -m phyglobt.preprocessing --ais_dir <raw_ais_dir> --out_dir <scenes_dir> --domain US
python -m phyglobt.preprocessing --ais_dir <raw_ais_dir> --out_dir <scenes_dir> --domain DM
```

3. Train/evaluate with the scene directory:

```bash
python -m phyglobt.train --data_dir <scenes_dir> --save_dir ./checkpoints --epochs 200 --ratios 1.0 0.5 0.1
```