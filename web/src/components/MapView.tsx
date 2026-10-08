import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { AssociationResult, SceneDetail } from '../lib/types';

export interface MapViewHandle {
  focusScene: (s: SceneDetail) => void;
}

interface Props {
  scene: SceneDetail | null;
  result: AssociationResult | null;
}

const SENSOR_A_COLOR = '#3b82f6';
const SENSOR_B_COLOR = '#ef4444';
const TP_COLOR = '#22c55e'; // prediction matches a ground-truth pair
const FP_COLOR = '#f59e0b'; // prediction with no ground-truth counterpart
const GT_COLOR = '#cbd5e1'; // neutral slate — GT endpoints are uniform; pairing is shown on hover
const HOVER_COLOR = '#22d3ee'; // bright cyan — hovered pair highlight (line + popup)

const M_PER_DEG_LAT = 111320.0;

/** Basemap options, ordered by preference. 'esri' is the default because it is
 *  WGS84 (matches AIS tracks) and reachable from CN networks, where OSM/Carto
 *  tiles frequently time out. AMap is GCJ-02 (~500 m offset) — labelled as such. */
interface BasemapDef {
  id: string;
  label: string;
  tiles?: string[];
  attribution?: string;
  maxzoom?: number;
}
const BASEMAPS: BasemapDef[] = [
  {
    id: 'esri',
    label: 'Esri Satellite (WGS84)',
    tiles: [
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    ],
    attribution: 'Imagery © Esri',
    maxzoom: 18,
  },
  {
    id: 'amap',
    label: 'AMap Roads (GCJ-02, offset)',
    tiles: [1, 2, 3, 4].map(
      (n) =>
        `https://webrd0${n}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}`,
    ),
    attribution: '© AMap',
    maxzoom: 18,
  },
  {
    id: 'osm',
    label: 'OpenStreetMap',
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    attribution: '© OpenStreetMap contributors',
    maxzoom: 19,
  },
  { id: 'none', label: 'None (dark)' },
];

/**
 * Ground-truth pair endpoints are drawn in ONE neutral colour (GT_COLOR).
 * A unique hue per pair was tried originally (golden-angle spread), but with
 * 100+ pairs the pigeonhole principle makes hues indistinguishable (< 2.5°
 * apart), so colour carried no information and read as noise. Pair identity
 * is communicated by hovering instead: the hovered pair's endpoints +
 * connecting line light up in HOVER_COLOR and a popup shows the MMSIs.
 */
interface PairEndpoint {
  kind: 'gt' | 'pred';
  idx: number;
  color: string;
  a: [number, number]; // [lon, lat] endpoint of the sensor-A track
  b: [number, number]; // [lon, lat] endpoint of the sensor-B track
  aId: number;
  bId: number;
  aMmsi: string;
  bMmsi: string;
}

/** Approximate a radius (meters) as a lat/lon ring centered at (lon, lat). */
function circlePolygon(lon: number, lat: number, radiusM: number, segments = 64): [number, number][] {
  const dLat = radiusM / M_PER_DEG_LAT;
  const dLon = radiusM / (M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180));
  const pts: [number, number][] = [];
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * 2 * Math.PI;
    pts.push([lon + dLon * Math.cos(a), lat + dLat * Math.sin(a)]);
  }
  pts.push(pts[0]!); // close the ring
  return pts;
}

/** Hover popup content for a GT / prediction pair. */
function pairPopupHtml(pair: PairEndpoint): string {
  const label = (sensor: string, id: number, mmsi: string) =>
    mmsi ? `Sensor ${sensor} · #${id} · MMSI <b>${mmsi}</b>` : `Sensor ${sensor} · #${id}`;
  let title: string;
  if (pair.kind === 'gt') {
    title = 'Ground-truth pair';
  } else {
    const tp = pair.color === TP_COLOR;
    title = `<span class="${tp ? 'tp' : 'fp'}">Prediction — ${tp ? 'TP' : 'FP'}</span>`;
  }
  return (
    `<div class="pair-popup">` +
    `<div class="pair-popup-title">${title}</div>` +
    `<div>${label('A', pair.aId, pair.aMmsi)}</div>` +
    `<div>${label('B', pair.bId, pair.bMmsi)}</div>` +
    `</div>`
  );
}

function makeGeoJson(scene: SceneDetail, gtPairs: PairEndpoint[], predPairs: PairEndpoint[]) {
  const features: GeoJSON.Feature[] = [];

  for (const track of scene.tracks) {
    const coords: [number, number][] = track.points.map((p) => [p.lon, p.lat]);
    features.push({
      type: 'Feature',
      properties: {
        type: 'track',
        sensor: track.sensor,
        trackId: track.id,
        color: track.sensor === 'A' ? SENSOR_A_COLOR : SENSOR_B_COLOR,
      },
      geometry: { type: 'LineString', coordinates: coords },
    });
  }

  // Ground truth: neutral endpoint markers (uniform colour). Hovering a
  // marker highlights the pair (cyan line + popup with MMSIs) — no permanent
  // lines or rainbow colours, which get unreadable in dense scenes.
  for (const pair of gtPairs) {
    features.push({
      type: 'Feature',
      properties: { type: 'gt-point', kind: 'gt', pairIdx: pair.idx, color: pair.color },
      geometry: { type: 'Point', coordinates: pair.a },
    });
    features.push({
      type: 'Feature',
      properties: { type: 'gt-point', kind: 'gt', pairIdx: pair.idx, color: pair.color },
      geometry: { type: 'Point', coordinates: pair.b },
    });
  }

  // Predictions: endpoint RINGS (not connecting lines). Green ring = true
  // positive (the pair exists in GT), orange ring = false match. Hovering a
  // ring draws the connecting line.
  for (const pair of predPairs) {
    features.push({
      type: 'Feature',
      properties: { type: 'pred-point', kind: 'pred', pairIdx: pair.idx, color: pair.color },
      geometry: { type: 'Point', coordinates: pair.a },
    });
    features.push({
      type: 'Feature',
      properties: { type: 'pred-point', kind: 'pred', pairIdx: pair.idx, color: pair.color },
      geometry: { type: 'Point', coordinates: pair.b },
    });
  }

  return { type: 'FeatureCollection', features } as GeoJSON.FeatureCollection;
}

function makeSensorsGeoJson(scene: SceneDetail | null) {
  const markers: GeoJSON.Feature[] = [];
  const ranges: GeoJSON.Feature[] = [];
  for (const s of scene?.sensors ?? []) {
    const color = s.id === 1 ? SENSOR_A_COLOR : SENSOR_B_COLOR;
    markers.push({
      type: 'Feature',
      properties: { type: 'sensor', sensor: s.id, color },
      geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
    });
    if (s.max_range_m && s.max_range_m > 0) {
      ranges.push({
        type: 'Feature',
        properties: { type: 'sensor-range', sensor: s.id, color },
        geometry: {
          type: 'Polygon',
          coordinates: [circlePolygon(s.lon, s.lat, s.max_range_m)],
        },
      });
    }
  }
  return {
    markers: { type: 'FeatureCollection', features: markers } as GeoJSON.FeatureCollection,
    ranges: { type: 'FeatureCollection', features: ranges } as GeoJSON.FeatureCollection,
  };
}

interface LayerVis {
  tracks: boolean;
  gt: boolean;
  pred: boolean;
  sensors: boolean;
  labels: boolean;
}

interface LayerStackData {
  gj: GeoJSON.FeatureCollection | null;
  sensorsGj: ReturnType<typeof makeSensorsGeoJson>;
  gtPairs: PairEndpoint[];
  predPairs: PairEndpoint[];
  vis: LayerVis;
  basemap: string;
}

const VECTOR_LAYER_ORDER = [
  'track-lines',
  'gt-pair-points',
  'pred-pair-points',
  'pair-hover-line',
  'sensor-range-fill',
  'sensor-range-outline',
  'sensor-dots',
];

function firstVectorLayer(map: maplibregl.Map): string | undefined {
  return VECTOR_LAYER_ORDER.find((id) => map.getLayer(id));
}

/** Idempotently (re)build the full vector layer stack on the map.
 *
 * MUST only be called after the map's `load` event has fired (tracked by
 * styleReadyRef), otherwise addSource/addLayer throw "Style is not done
 * loading". Any style/map errors are swallowed so a transient failure (e.g. a
 * map instance being recreated during a hot reload) cannot crash the React tree.
 */
function buildLayerStack(
  map: maplibregl.Map,
  data: LayerStackData,
  hoverHandlersRef: React.MutableRefObject<
    { layer: string; move: (e: maplibregl.MapLayerMouseEvent) => void; leave: () => void }[]
  >,
  popup: maplibregl.Popup,
) {
  try {
    const { gj, sensorsGj } = data;

    for (const id of VECTOR_LAYER_ORDER) {
      if (map.getLayer(id)) map.removeLayer(id);
    }
    for (const src of ['tracks', 'pair-hover', 'sensor-markers', 'sensor-ranges']) {
      if (map.getSource(src)) map.removeSource(src);
    }

    // Un-bind previous hover handlers before re-adding their layers.
    for (const h of hoverHandlersRef.current) {
      map.off('mousemove', h.layer, h.move);
      map.off('mouseleave', h.layer, h.leave);
    }
    hoverHandlersRef.current = [];

    if (!gj) return;

    map.addSource('tracks', { type: 'geojson', data: gj });
    map.addLayer({
      id: 'track-lines',
      type: 'line',
      source: 'tracks',
      filter: ['==', ['get', 'type'], 'track'],
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 2,
        'line-opacity': 0.9,
      },
    });

    // Ground-truth pair endpoint markers (uniform neutral colour; pair identity
    // is revealed by hovering — see pairPopupHtml).
    map.addLayer({
      id: 'gt-pair-points',
      type: 'circle',
      source: 'tracks',
      filter: ['==', ['get', 'type'], 'gt-point'],
      paint: {
        'circle-radius': 4.5,
        'circle-color': GT_COLOR,
        'circle-stroke-color': '#ffffff',
        'circle-stroke-width': 1.2,
        'circle-opacity': 0.9,
      },
    });

    // Prediction pair endpoint RINGS: hollow circles, green = TP, orange = FP.
    map.addLayer({
      id: 'pred-pair-points',
      type: 'circle',
      source: 'tracks',
      filter: ['==', ['get', 'type'], 'pred-point'],
      paint: {
        'circle-radius': 7,
        'circle-opacity': 0, // hollow: stroke only, so GT dots stay visible beneath
        'circle-stroke-color': ['get', 'color'],
        'circle-stroke-width': 2.2,
        'circle-stroke-opacity': 0.95,
      },
    });

    // Hover-only: the single association line for the hovered pair (GT or pred).
    map.addSource('pair-hover', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });
    map.addLayer({
      id: 'pair-hover-line',
      type: 'line',
      source: 'pair-hover',
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 2.5,
        'line-opacity': 1,
      },
    });

    // Sensor positions + detection range circles.
    // Paint order matters: the range fill must be added BEFORE the dots, or its
    // 8%-opacity wash covers the 7px sensor markers (later layers paint on top).
    if (sensorsGj.ranges.features.length > 0) {
      map.addSource('sensor-ranges', { type: 'geojson', data: sensorsGj.ranges });
      map.addLayer({
        id: 'sensor-range-fill',
        type: 'fill',
        source: 'sensor-ranges',
        paint: {
          'fill-color': ['get', 'color'],
          'fill-opacity': 0.08,
        },
      });
      map.addLayer({
        id: 'sensor-range-outline',
        type: 'line',
        source: 'sensor-ranges',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 1.5,
          'line-dasharray': [4, 2],
          'line-opacity': 0.55,
        },
      });
    }
    if (sensorsGj.markers.features.length > 0) {
      map.addSource('sensor-markers', { type: 'geojson', data: sensorsGj.markers });
      map.addLayer({
        id: 'sensor-dots',
        type: 'circle',
        source: 'sensor-markers',
        paint: {
          'circle-radius': 7,
          'circle-color': ['get', 'color'],
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2,
        },
      });
    }

    // Hover: highlight the hovered pair (cyan connecting line) and show a
    // popup with the track ids and MMSIs. Works for GT markers and
    // prediction rings alike.
    const pairOf = (feat: GeoJSON.Feature | undefined): PairEndpoint | undefined => {
      const kind = feat?.properties?.kind as 'gt' | 'pred' | undefined;
      const idx = feat?.properties?.pairIdx as number | undefined;
      if (kind === undefined || idx === undefined) return undefined;
      const list = kind === 'gt' ? data.gtPairs : data.predPairs;
      return list.find((p) => p.idx === idx);
    };
    const move = (e: maplibregl.MapLayerMouseEvent) => {
      const pair = pairOf(e.features?.[0]);
      const src = map.getSource('pair-hover');
      if (pair && src) {
        (src as maplibregl.GeoJSONSource).setData({
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: { color: HOVER_COLOR },
              geometry: { type: 'LineString', coordinates: [pair.a, pair.b] },
            },
          ],
        });
        popup
          .setLngLat(e.lngLat)
          .setHTML(pairPopupHtml(pair))
          .addTo(map);
        map.getCanvas().style.cursor = 'pointer';
      }
    };
    const leave = () => {
      const src = map.getSource('pair-hover');
      if (src) {
        (src as maplibregl.GeoJSONSource).setData({
          type: 'FeatureCollection',
          features: [],
        });
      }
      popup.remove();
      map.getCanvas().style.cursor = '';
    };
    const hoverLayers = ['gt-pair-points', 'pred-pair-points'];
    for (const layer of hoverLayers) {
      map.on('mousemove', layer, move);
      map.on('mouseleave', layer, leave);
      hoverHandlersRef.current.push({ layer, move, leave });
    }

    applyVisibility(map, data.vis);
  } catch (e) {
    console.error('[MapView] failed to build layer stack:', e);
  }
}

function applyVisibility(map: maplibregl.Map, vis: LayerVis) {
  const set = (id: string, on: boolean) => {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
    }
  };
  set('track-lines', vis.tracks);
  set('gt-pair-points', vis.gt);
  set('pred-pair-points', vis.pred);
  set('pair-hover-line', vis.gt || vis.pred);
  set('sensor-range-fill', vis.sensors);
  set('sensor-range-outline', vis.sensors);
  set('sensor-dots', vis.sensors);
}

/** Swap the raster basemap; inserted BELOW all vector layers. */
function setBasemap(map: maplibregl.Map, id: string) {
  try {
    if (map.getLayer('basemap-raster')) map.removeLayer('basemap-raster');
    if (map.getSource('basemap')) map.removeSource('basemap');
    const bm = BASEMAPS.find((b) => b.id === id);
    if (bm?.tiles?.length) {
      map.addSource('basemap', {
        type: 'raster',
        tiles: bm.tiles,
        tileSize: 256,
        maxzoom: bm.maxzoom,
        attribution: bm.attribution,
      });
      map.addLayer({ id: 'basemap-raster', type: 'raster', source: 'basemap' }, firstVectorLayer(map));
    }
  } catch (e) {
    console.warn('[MapView] basemap switch failed:', e);
  }
}

const MapView = forwardRef<MapViewHandle, Props>(function MapView({ scene, result }, ref) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  // One-shot flag: has the style's `load` event fired for THIS map instance?
  const styleReadyRef = useRef(false);
  // Latest layer-stack data, so the one-shot `load` handler can build with
  // whatever the scene/result state happens to be at load time.
  const dataRef = useRef<LayerStackData>({
    gj: null,
    sensorsGj: makeSensorsGeoJson(null),
    gtPairs: [],
    predPairs: [],
    vis: { tracks: true, gt: true, pred: true, sensors: true, labels: false },
    basemap: BASEMAPS[0]!.id,
  });
  // Shared hover popup (pair highlight + MMSI info).
  const popupRef = useRef(
    new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      offset: 14,
      maxWidth: '320px',
      className: 'pair-popup-anchor',
    }),
  );
  // DOM markers for MMSI labels (a symbol layer would need remote font
  // glyphs; DOM markers work fully offline).
  const labelsRef = useRef<maplibregl.Marker[]>([]);
  // Hover handlers must be re-bound whenever their closure (pairs, map)
  // changes. We store the current bound fns so the next run can un-bind them.
  const hoverHandlersRef = useRef<
    { layer: string; move: (e: maplibregl.MapLayerMouseEvent) => void; leave: () => void }[]
  >([]);

  const [basemap, setBasemapId] = useState<string>(BASEMAPS[0]!.id);
  const [vis, setVis] = useState<LayerVis>({ tracks: true, gt: true, pred: true, sensors: true, labels: false });

  const gtPairs = useMemo<PairEndpoint[]>(() => {
    const trackById = new Map(scene?.tracks.map((t) => [t.id, t]) ?? []);
    const pairs: PairEndpoint[] = [];
    for (const [i, m] of (scene?.gt ?? []).entries()) {
      const ta = trackById.get(m.track_a);
      const tb = trackById.get(m.track_b);
      if (!ta || !tb || ta.points.length === 0 || tb.points.length === 0) continue;
      const pa = ta.points[ta.points.length - 1]!;
      const pb = tb.points[tb.points.length - 1]!;
      pairs.push({
        kind: 'gt',
        idx: i,
        color: GT_COLOR,
        a: [pa.lon, pa.lat],
        b: [pb.lon, pb.lat],
        aId: ta.id,
        bId: tb.id,
        aMmsi: ta.mmsi ?? '',
        bMmsi: tb.mmsi ?? '',
      });
    }
    return pairs;
  }, [scene]);

  const predPairs = useMemo<PairEndpoint[]>(() => {
    if (!scene || !result) return [];
    const gtKey = new Set((scene.gt ?? []).map((m) => `${m.track_a}-${m.track_b}`));
    const trackById = new Map(scene.tracks.map((t) => [t.id, t]));
    const pairs: PairEndpoint[] = [];
    for (const [i, m] of result.matches.entries()) {
      const ta = trackById.get(m.track_a);
      const tb = trackById.get(m.track_b);
      if (!ta || !tb || ta.points.length === 0 || tb.points.length === 0) continue;
      const tp = gtKey.has(`${m.track_a}-${m.track_b}`);
      const pa = ta.points[ta.points.length - 1]!;
      const pb = tb.points[tb.points.length - 1]!;
      pairs.push({
        kind: 'pred',
        idx: i,
        color: tp ? TP_COLOR : FP_COLOR,
        a: [pa.lon, pa.lat],
        b: [pb.lon, pb.lat],
        aId: ta.id,
        bId: tb.id,
        aMmsi: ta.mmsi ?? '',
        bMmsi: tb.mmsi ?? '',
      });
    }
    return pairs;
  }, [scene, result]);

  useImperativeHandle(ref, () => ({
    focusScene(s: SceneDetail) {
      const map = mapRef.current;
      if (!map) return;
      const bounds = new maplibregl.LngLatBounds();
      for (const t of s.tracks) {
        for (const p of t.points) bounds.extend([p.lon, p.lat]);
      }
      // Also fit the radar sites: a 30–40 km detection ring is far wider than
      // the track cluster, so the sensor markers themselves can sit outside a
      // track-only fit and the user would never see where the radars are.
      for (const sensor of s.sensors ?? []) {
        bounds.extend([sensor.lon, sensor.lat]);
      }
      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, { padding: 60, maxZoom: 14, duration: 600 });
      }
    },
  }));

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    // The inline style starts out with ONLY a local background layer. If the
    // basemap tiles are unreachable, the style still finishes loading, so
    // `load` reliably fires and the vector layers can be added.
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          { id: 'base-bg', type: 'background', paint: { 'background-color': '#0f172a' } },
        ],
      },
      center: [-90, 28],
      zoom: 4.5,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
    mapRef.current = map;
    // Dev-only hook: lets Playwright/QA assert layer state programmatically.
    if (import.meta.env.DEV) {
      (window as unknown as Record<string, unknown>).__phyglobtMap = map;
    }
    map.on('load', () => {
      styleReadyRef.current = true;
      // Basemap first (below vector layers); tile failures only cost the
      // basemap image — the vector layers are unaffected.
      setBasemap(map, dataRef.current.basemap);
      buildLayerStack(map, dataRef.current, hoverHandlersRef, popupRef.current);
    });
    return () => {
      styleReadyRef.current = false;
      popupRef.current.remove();
      for (const m of labelsRef.current) m.remove();
      labelsRef.current = [];
      map.remove();
      mapRef.current = null;
    };
  }, []);

  const gj = useMemo(
    () => (scene ? makeGeoJson(scene, gtPairs, predPairs) : null),
    [scene, gtPairs, predPairs],
  );

  const sensorsGj = useMemo(() => makeSensorsGeoJson(scene), [scene]);

  // Rebuild the vector layers whenever scene/result change. The map instance
  // can be recreated independently (e.g. React StrictMode double mount), so
  // the build is gated on styleReadyRef — a one-shot flag flipped by the map's
  // own `load` event. (Do NOT gate on map.isStyleLoaded(): an unreachable
  // raster basemap keeps it false forever.)
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    dataRef.current = {
      ...dataRef.current,
      gj,
      sensorsGj,
      gtPairs,
      predPairs,
    };
    if (styleReadyRef.current) {
      buildLayerStack(map, dataRef.current, hoverHandlersRef, popupRef.current);
    }
  }, [gj, sensorsGj, gtPairs, predPairs]);

  // MMSI labels: one DOM marker at each track's endpoint. Kept off by default
  // (300+ labels is a lot); the "MMSI labels" toggle opts in. DOM markers are
  // used instead of a symbol layer so no remote font glyphs are required.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    for (const m of labelsRef.current) m.remove();
    labelsRef.current = [];
    if (!scene || !vis.labels) return;
    for (const t of scene.tracks) {
      if (t.points.length === 0) continue;
      const p = t.points[t.points.length - 1]!;
      const el = document.createElement('div');
      el.className = 'track-label' + (t.sensor === 'B' ? ' track-label-b' : '');
      el.textContent = t.mmsi ? `MMSI ${t.mmsi}` : `#${t.id}`;
      const marker = new maplibregl.Marker({ element: el, anchor: 'left', offset: [8, 0] })
        .setLngLat([p.lon, p.lat])
        .addTo(map);
      labelsRef.current.push(marker);
    }
    return () => {
      for (const m of labelsRef.current) m.remove();
      labelsRef.current = [];
    };
  }, [scene, vis.labels]);

  useEffect(() => {
    dataRef.current.vis = vis;
    const map = mapRef.current;
    if (map && styleReadyRef.current) applyVisibility(map, vis);
  }, [vis]);

  useEffect(() => {
    dataRef.current.basemap = basemap;
    const map = mapRef.current;
    if (map && styleReadyRef.current) setBasemap(map, basemap);
  }, [basemap]);

  const sensors = scene?.sensors ?? [];
  const errModel = scene?.error_model ?? null;

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map-container" />
      {!scene && <div className="map-placeholder">Select a scene to visualize its tracks.</div>}

      <div className="map-controls">
        <label className="ctl-basemap">
          <span>Basemap</span>
          <select value={basemap} onChange={(e) => setBasemapId(e.target.value)}>
            {BASEMAPS.map((b) => (
              <option key={b.id} value={b.id}>
                {b.label}
              </option>
            ))}
          </select>
        </label>
        <div className="ctl-toggles">
          {(
            [
              ['tracks', 'Tracks'],
              ['gt', 'GT pairs'],
              ['pred', 'Predictions'],
              ['sensors', 'Sensors & range'],
              ['labels', 'MMSI labels'],
            ] as [keyof LayerVis, string][]
          ).map(([key, label]) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={vis[key]}
                onChange={(e) => setVis((v) => ({ ...v, [key]: e.target.checked }))}
              />
              {label}
            </label>
          ))}
        </div>
      </div>

      {scene && (
        <>
          <div className="sensor-info">
            {sensors.length > 0 && (
              <div className="sensor-info-rows">
                {sensors.map((s) => (
                  <div key={s.id} className="sensor-info-row">
                    <i style={{ background: s.id === 1 ? SENSOR_A_COLOR : SENSOR_B_COLOR }} />
                    Sensor {s.id}: {s.lat.toFixed(4)}°, {s.lon.toFixed(4)}° · max range{' '}
                    {s.max_range_m ? `${(s.max_range_m / 1000).toFixed(0)} km` : '—'}
                  </div>
                ))}
              </div>
            )}
            {errModel && (
              <div className="sensor-info-noise">
                Radar noise ({errModel.difficulty}): σρ = {errModel.sigma_rho_m} m ·
                σθ = {errModel.sigma_theta_deg}° (at 10 km, ∝ (r/10km)<sup>0.8</sup>) ·
                miss {errModel.drop_rate} · overlap {errModel.window_overlap}
              </div>
            )}
          </div>
          <div className="legend">
            <span><i style={{ background: SENSOR_A_COLOR }} /> Sensor A</span>
            <span><i style={{ background: SENSOR_B_COLOR }} /> Sensor B</span>
            <span><i className="legend-dot" style={{ background: GT_COLOR }} /> GT pair endpoint</span>
            <span><i className="legend-dot" style={{ background: HOVER_COLOR }} /> Hovered pair</span>
            {result && (
              <>
                <span><i className="legend-ring" style={{ borderColor: TP_COLOR }} /> TP</span>
                <span><i className="legend-ring" style={{ borderColor: FP_COLOR }} /> FP</span>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
});

export default MapView;
