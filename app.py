from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch

from utils.detection import DetectorSuite
from utils.geolocation import pixel_to_gps
from utils.tracking import CandidateManager
from utils.triage import to_geojson
from utils.hazard_detection import HazardDetector
from utils.route_planning import plan_visit_order


ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="RescueVision AI — Disaster Detection and Rescue Support",
    page_icon="🚁",
    layout="wide",
)

st.title("RescueVision AI")
st.subheader("AI-assisted Disaster Detection and Search Support")
st.caption(
    "Detect people, fire and smoke in drone imagery. "
    "All detections and location estimates require human verification."
)


@st.cache_resource(show_spinner="Loading person detection models...")
def load_detectors() -> DetectorSuite:
    return DetectorSuite(ROOT, DEVICE)


@st.cache_resource(show_spinner="Loading fire and smoke model...")
def load_hazard_detector() -> HazardDetector:
    return HazardDetector(ROOT / "models" / "fire_smoke.pt", DEVICE)


with st.sidebar:
    st.header("Drone / Flight Telemetry")
    st.caption(
        "Enter the drone's actual GPS position. The default coordinates are "
        "examples only; replace them with the flight telemetry for this footage."
    )
    drone_lat = st.number_input(
        "Drone latitude", min_value=-90.0, max_value=90.0,
        value=13.0827, format="%.6f",
    )
    drone_lon = st.number_input(
        "Drone longitude", min_value=-180.0, max_value=180.0,
        value=80.2707, format="%.6f",
    )
    altitude = st.number_input("Altitude above ground (m)", min_value=1.0, value=50.0)
    yaw = st.number_input(
        "Drone heading / yaw (degrees)", min_value=0.0,
        max_value=360.0, value=0.0,
    )
    process_every = st.slider("Process every N frames", 1, 8, 2)

    st.divider()
    st.header("Rescue Route Start")
    start_mode = st.radio(
        "Choose starting point",
        ["Rescue team GPS location", "Manual entry-point coordinates"],
        help="Both options require latitude/longitude coordinates.",
    )
    if start_mode == "Rescue team GPS location":
        start_lat = st.number_input(
            "Rescue team latitude", min_value=-90.0, max_value=90.0,
            value=float(drone_lat), format="%.6f",
        )
        start_lon = st.number_input(
            "Rescue team longitude", min_value=-180.0, max_value=180.0,
            value=float(drone_lon), format="%.6f",
        )
    else:
        start_lat = st.number_input(
            "Manual entry-point latitude", min_value=-90.0, max_value=90.0,
            value=float(drone_lat), format="%.6f",
        )
        start_lon = st.number_input(
            "Manual entry-point longitude", min_value=-180.0, max_value=180.0,
            value=float(drone_lon), format="%.6f",
        )

    st.caption(f"Compute device: {'CUDA' if DEVICE == 0 else 'CPU'}")


uploaded_file = st.file_uploader(
    "Upload drone image / video footage",
    type=["jpg", "jpeg", "png", "mp4", "m4v", "mov", "avi", "mkv"],
)

try:
    detectors = load_detectors()
except Exception as exc:
    detectors = None
    st.error(f"Person detection model initialization failed: {exc}")

try:
    hazard_detector = load_hazard_detector()
except Exception as exc:
    hazard_detector = None
    st.error(f"Fire/smoke detection model initialization failed: {exc}")

if detectors is not None:
    if detectors.custom_available:
        st.success(f"Custom SAR person detector active: {detectors.primary_path.name}")
    else:
        st.warning("Custom SAR detector unavailable; using COCO person fallback.")
    if detectors.load_error:
        st.info(detectors.load_error)

if hazard_detector is not None:
    st.success("Fire and smoke detector active: models/fire_smoke.pt")
else:
    st.warning("Fire/smoke detector unavailable; hazard analysis will be skipped.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("People Detected", 0)
m2.metric("Tracked Candidates", 0)
m3.metric("Frames Processed", 0)
m4.metric("System Status", "READY")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Annotated Drone Image / Video")
    image_placeholder = st.empty()
with right:
    st.subheader("Rescue Map and Search Route")
    map_placeholder = st.empty()

alert_placeholder = st.empty()


def draw_map_and_route(
    placeholder,
    start: tuple[float, float],
    route: list[dict[str, Any]],
) -> None:
    if not route:
        placeholder.info(
            "No route targets yet. Run analysis and ensure people have been detected."
        )
        return

    path_coords = [[float(start[1]), float(start[0])]]
    point_data = [{
        "position": [float(start[1]), float(start[0])],
        "label": "START",
        "kind": "Rescue start",
    }]
    for index, target in enumerate(route, start=1):
        lat = float(target["latitude"])
        lon = float(target["longitude"])
        path_coords.append([lon, lat])
        point_data.append({
            "position": [lon, lat],
            "label": f"Person {index}",
            "kind": str(target.get("label", "Detected person")),
        })

    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": path_coords, "name": "Estimated search route"}],
            get_path="path",
            get_color=[30, 150, 255],
            width_min_pixels=4,
            get_width=5,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="position",
            get_radius=7,
            get_fill_color=[220, 60, 60],
            pickable=True,
        ),
    ]
    view = pdk.ViewState(
        latitude=float(start[0]),
        longitude=float(start[1]),
        zoom=16,
        pitch=0,
    )
    placeholder.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view,
            tooltip={"text": "{label}: {kind}"},
            map_style=None,
        ),
        use_container_width=True,
    )


def process_frame(
    frame: np.ndarray,
    detectors: DetectorSuite,
    hazard_detector: HazardDetector | None,
    manager: CandidateManager,
    width: int,
    height: int,
    drone_lat: float,
    drone_lon: float,
    altitude: float,
    yaw: float,
    alert_placeholder,
) -> tuple[np.ndarray, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    annotated = frame.copy()
    detections, rejected = detectors.process(frame)
    hazards = hazard_detector.process(frame) if hazard_detector else []
    people_locations: list[dict[str, Any]] = []

    for detection in detections:
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        lat, lon = pixel_to_gps(
            bbox, width, height, drone_lat, drone_lon, altitude, yaw
        )
        source = detection.get("source", "PERSON")
        track_id = detection.get("track_id")

        # Tiled predictions have no persistent tracker ID. Keep them as
        # individual detections; do not merge all None IDs into one candidate.
        candidate_id = None
        created = False
        if track_id is not None:
            crop = frame[y1:y2, x1:x2]
            evidence = (
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                if crop.size else None
            )
            candidate_id, created = manager.update(
                detection, lat, lon, evidence
            )

        people_locations.append({
            "latitude": float(lat),
            "longitude": float(lon),
            "confidence": float(confidence),
            "source": source,
            "track_id": track_id,
            "candidate_id": candidate_id,
            "label": f"{source} {confidence:.2f}",
        })

        color = (40, 220, 120) if "SAR" in source else (255, 210, 40)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{candidate_id or source} {confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

        if created and candidate_id:
            priority = manager.candidates[candidate_id]["priority"]
            alert_placeholder.warning(
                f"POSSIBLE SURVIVOR {candidate_id} — {priority} PRIORITY. "
                "Human verification required."
            )

    for hazard in hazards:
        hx1, hy1, hx2, hy2 = [int(v) for v in hazard["bbox"]]
        hx1, hy1 = max(0, hx1), max(0, hy1)
        hx2, hy2 = min(width, hx2), min(height, hy2)
        name = hazard["class_name"].upper()
        confidence = float(hazard["confidence"])
        color = (0, 0, 255) if name == "FIRE" else (255, 100, 0)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 3)
        cv2.putText(
            annotated, f"{name} {confidence:.2f}",
            (hx1, max(20, hy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2,
        )

    return annotated, len(detections), rejected, hazards, people_locations


if uploaded_file is not None and st.button(
    "START DISASTER ANALYSIS", type="primary", use_container_width=True
):
    if detectors is None:
        st.error("Person detector could not be loaded.")
    else:
        suffix = Path(uploaded_file.name).suffix.lower()
        is_image = suffix in [".jpg", ".jpeg", ".png"]
        manager = CandidateManager()

        processed_frames = 0
        total_person_detections = 0
        rejected_vehicles = 0
        fire_count = 0
        smoke_count = 0
        image_people: list[dict[str, Any]] = []
        temp_path: str | None = None
        capture = None
        completed = False

        try:
            if is_image:
                data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Could not decode this image.")
                else:
                    h, w = frame.shape[:2]
                    annotated, count, rejected, hazards, image_people = process_frame(
                        frame, detectors, hazard_detector, manager, w, h,
                        drone_lat, drone_lon, altitude, yaw, alert_placeholder,
                    )
                    processed_frames = 1
                    total_person_detections = count
                    rejected_vehicles = rejected
                    fire_count = sum(x["class_name"].lower() == "fire" for x in hazards)
                    smoke_count = sum(x["class_name"].lower() == "smoke" for x in hazards)
                    image_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_container_width=True,
                    )
                    completed = True
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                capture = cv2.VideoCapture(temp_path)
                if not capture.isOpened():
                    st.error("Could not open the uploaded video.")
                else:
                    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = st.progress(0)

                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        if frame_number % process_every != 0:
                            continue

                        annotated, count, rejected, hazards, people = process_frame(
                            frame, detectors, hazard_detector, manager, w, h,
                            drone_lat, drone_lon, altitude, yaw, alert_placeholder,
                        )
                        processed_frames += 1
                        total_person_detections += count
                        rejected_vehicles += rejected
                        fire_count += sum(x["class_name"].lower() == "fire" for x in hazards)
                        smoke_count += sum(x["class_name"].lower() == "smoke" for x in hazards)

                        image_placeholder.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True,
                        )
                        if total_frames > 0:
                            progress.progress(min(frame_number / total_frames, 1.0))

                    progress.empty()
                    completed = processed_frames > 0

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    st.warning("Could not remove temporary video file; results remain available.")

        if completed:
            m1.metric("People Detected", total_person_detections)
            m2.metric("Tracked Candidates", len(manager.candidates))
            m3.metric("Frames Processed", processed_frames)
            m4.metric("System Status", "COMPLETE")

            st.subheader("Hazard Detection")
            hc1, hc2 = st.columns(2)
            hc1.metric("Fire detections", fire_count)
            hc2.metric("Smoke detections", smoke_count)
            st.caption(
                "Hazard totals count detections per processed frame, not unique incidents."
            )

            # Single-image detections are displayed even though they are not
            # repeated-track-confirmed candidates.
            if is_image:
                st.subheader("People detected in this image")
                st.write(f"Detected person instances: **{len(image_people)}**")
                if image_people:
                    display_rows = [
                        {
                            "Detection": i + 1,
                            "Source": p["source"],
                            "Confidence": round(p["confidence"], 3),
                            "Estimated latitude": round(p["latitude"], 6),
                            "Estimated longitude": round(p["longitude"], 6),
                        }
                        for i, p in enumerate(image_people)
                    ]
                    st.dataframe(display_rows, use_container_width=True, hide_index=True)

            st.subheader("Rescue Search Route")
            # Use all individual detections for a still image. For video,
            # use only candidates confirmed by repeated tracking.
            if is_image:
                route_targets = image_people
            else:
                route_targets = [
                    {
                        "latitude": c["latitude"],
                        "longitude": c["longitude"],
                        "label": candidate_id,
                        "confidence": c["max_confidence"],
                    }
                    for candidate_id, c in manager.candidates.items()
                ]

            # Remove near-identical points to avoid repeated route stops.
            unique_targets = []
            seen = set()
            for item in route_targets:
                key = (round(float(item["latitude"]), 6), round(float(item["longitude"]), 6))
                if key not in seen:
                    seen.add(key)
                    unique_targets.append(item)

            start = (float(start_lat), float(start_lon))
            if unique_targets:
                route, distance_m = plan_visit_order(start, unique_targets)
                draw_map_and_route(map_placeholder, start, route)
                st.metric("Estimated straight-line route distance", f"{distance_m:.1f} m")
                st.write("**Recommended visit order:**")
                for i, target in enumerate(route, start=1):
                    st.write(
                        f"{i}. {target.get('label', 'Detected person')} — "
                        f"{float(target['latitude']):.6f}, "
                        f"{float(target['longitude']):.6f}"
                    )
            else:
                map_placeholder.info(
                    "No route targets were found. A route cannot be calculated without detected people."
                )

            st.warning(
                "Route limitation: this is an approximate straight-line waypoint order "
                "using pixel-to-GPS estimates. It does not know roads, terrain, water depth, "
                "building access or verified safe passages. Do not use it as a real-world "
                "dispatch route without independent map and responder verification."
            )

            st.subheader("Tracked Rescue Candidates")
            if not manager.candidates:
                st.info(
                    "No person tracks reached the repeated-observation threshold. "
                    "Single-image detections are still listed above."
                )
            for candidate_id, candidate in sorted(
                manager.candidates.items(),
                key=lambda item: item[1]["triage_score"],
                reverse=True,
            ):
                with st.expander(
                    f"{candidate_id} — {candidate['priority']} priority — "
                    f"{candidate['max_confidence']:.1%} confidence"
                ):
                    st.write(f"Source: {candidate['source']}")
                    st.write(f"Observations: {candidate['observations']}")
                    st.write(f"Triage score: {candidate['triage_score']:.2f}")
                    st.write(
                        f"Estimated location: {candidate['latitude']:.6f}, "
                        f"{candidate['longitude']:.6f}"
                    )

            st.download_button(
                "Download GeoJSON tracked candidates",
                data=to_geojson(manager.candidates),
                file_name="survivor_triage.geojson",
                mime="application/geo+json",
            )

            st.caption(
                f"Vehicle-like detections rejected: {rejected_vehicles}. "
                "All AI detections and route estimates require human verification."
            )
