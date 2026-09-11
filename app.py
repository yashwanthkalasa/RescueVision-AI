from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

from utils.detection import DetectorSuite
from utils.geolocation import pixel_to_gps
from utils.tracking import CandidateManager
from utils.triage import to_geojson

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="DisasterSight — AI Disaster Survivor Triage", page_icon="🚁", layout="wide")
st.title("DisasterSight — AI Disaster Survivor Triage")
st.subheader("From Drone Pixels to Rescue Decisions")
st.caption("AI-assisted search support for human responders. Every candidate requires human verification.")


@st.cache_resource(show_spinner="Loading detection models...")
def load_detectors() -> DetectorSuite:
    return DetectorSuite(ROOT, DEVICE)


with st.sidebar:
    st.header("Flight / Telemetry")
    drone_lat = st.number_input("Drone latitude", value=13.0827, format="%.6f")
    drone_lon = st.number_input("Drone longitude", value=80.2707, format="%.6f")
    altitude = st.number_input("Altitude above ground (m)", min_value=1.0, value=50.0)
    yaw = st.number_input("Drone heading / yaw (degrees)", min_value=0.0, max_value=360.0, value=0.0)
    process_every = st.slider("Process every N frames", min_value=1, max_value=8, value=2)
    st.info("Lower frame skipping improves detection and tracking but increases processing time.")
    st.caption(f"Compute device: {'CUDA' if DEVICE == 0 else 'CPU'}")

uploaded_video = st.file_uploader("Upload recorded drone / SAR footage", type=["mp4", "m4v", "mov", "avi", "mkv"])

try:
    detectors = load_detectors()
except Exception as exc:
    detectors = None
    st.error(f"Model initialization failed: {exc}")

if detectors is not None:
    if detectors.custom_available:
        st.success(f"Custom SAR detector active: {detectors.primary_path.name}")
    else:
        st.warning("Custom SAR model not found at models/best.pt. Running demonstration mode with the COCO person fallback.")
    if detectors.load_error:
        st.info(detectors.load_error)

metric_candidates, metric_observations, metric_frames, metric_status = st.columns(4)
metric_candidates.metric("Survivor Candidates", 0)
metric_observations.metric("Human Observations", 0)
metric_frames.metric("Frames Processed", 0)
metric_status.metric("System Status", "READY")

video_col, map_col = st.columns([1.35, 1])
with video_col:
    st.subheader("Annotated Drone Replay")
    video_placeholder = st.empty()
with map_col:
    st.subheader("Interactive Rescue Map")
    map_placeholder = st.empty()

alert_placeholder = st.empty()
run_analysis = uploaded_video is not None and st.button("START DISASTER ANALYSIS", type="primary", use_container_width=True)


def render_map(map_placeholder, candidates: dict) -> None:
    if not candidates:
        map_placeholder.info("Confirmed candidates will appear here after repeated observations.")
        return
    points = pd.DataFrame([{"lat": item["latitude"], "lon": item["longitude"]} for item in candidates.values()])
    map_placeholder.map(points, latitude="lat", longitude="lon", zoom=18)


if run_analysis and detectors is not None:
    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
    temp_path = None
    manager = CandidateManager()
    processed_frames = 0
    total_observations = 0
    rejected_vehicles = 0
    frame_number = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_path = temp_file.name
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            st.error("Could not open the uploaded video.")
            st.stop()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metric_status.metric("System Status", "ANALYSING")
        render_map(map_placeholder, manager.candidates)

        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            if frame_number % process_every != 0:
                continue
            processed_frames += 1
            annotated = frame.copy()
            detections, rejected = detectors.process(frame)
            rejected_vehicles += rejected
            for detection in detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                total_observations += 1
                x1, y1, x2, y2 = [max(0, int(value)) for value in bbox]
                x2, y2 = min(width, x2), min(height, y2)
                crop = frame[y1:y2, x1:x2]
                evidence = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size else None
                latitude, longitude = pixel_to_gps(bbox, width, height, drone_lat, drone_lon, altitude, yaw)
                candidate_id, created = manager.update(detection, latitude, longitude, evidence)
                color = (40, 220, 120) if detection["source"] == "SAR" else (255, 210, 40)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{candidate_id or detection['source']} {confidence:.2f}"
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if created and candidate_id:
                    priority = manager.candidates[candidate_id]["priority"]
                    alert_placeholder.error(f"POSSIBLE SURVIVOR {candidate_id} DETECTED — {priority} PRIORITY")
                    st.toast(f"Possible survivor {candidate_id} detected")
            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            metric_candidates.metric("Survivor Candidates", len(manager.candidates))
            metric_observations.metric("Human Observations", total_observations)
            metric_frames.metric("Frames Processed", processed_frames)
            render_map(map_placeholder, manager.candidates)
        capture.release()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    metric_status.metric("System Status", "COMPLETE")
    st.success("Flight analysis complete. AI recommendations remain subject to human verification.")
    st.caption(f"Vehicle-like detections rejected: {rejected_vehicles} | Custom detector: {'active' if detectors.custom_available else 'fallback only'}")
    st.divider()
    st.header("Prioritised Rescue Candidates")
    if not manager.candidates:
        st.warning("No repeated survivor candidate was confirmed. Manual search is still required.")
    for candidate_id, candidate in sorted(manager.candidates.items(), key=lambda item: item[1]["triage_score"], reverse=True):
        with st.expander(f"{candidate_id} — {candidate['priority']} PRIORITY — {candidate['max_confidence']:.1%} confidence", expanded=True):
            evidence_col, details_col = st.columns([1, 2])
            with evidence_col:
                if candidate["evidence"] is not None:
                    st.image(candidate["evidence"], caption=f"Best evidence crop for {candidate_id}", use_container_width=True)
            with details_col:
                st.write(f"**Detector:** {candidate['source']}  |  **Track:** {candidate['track_id']}")
                st.write(f"**Repeated observations:** {candidate['observations']}  |  **Triage score:** {candidate['triage_score']:.2f}")
                st.write(f"**Estimated location:** `{candidate['latitude']:.6f}, {candidate['longitude']:.6f}`")
                st.warning("AI recommendation only. Confirm visually and operationally before dispatch.")
    st.download_button("Download GeoJSON triage list", data=to_geojson(manager.candidates), file_name="survivor_triage.geojson", mime="application/geo+json")