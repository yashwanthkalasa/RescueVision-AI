"""Human-reviewable rescue priority and GeoJSON serialization."""

from __future__ import annotations

import json


def priority_score(confidence: float, observations: int) -> tuple[float, str]:
    score = 0.60 * confidence + 0.40 * min(observations / 10.0, 1.0)
    priority = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.50 else "LOW"
    return score, priority


def to_geojson(candidates: dict) -> str:
    features = []
    for candidate_id, candidate in candidates.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [candidate["longitude"], candidate["latitude"]]},
            "properties": {
                "candidate_id": candidate_id,
                "track_id": candidate["track_id"],
                "source": candidate["source"],
                "confidence": round(candidate["max_confidence"], 3),
                "observations": candidate["observations"],
                "triage_score": round(candidate["triage_score"], 3),
                "priority": candidate["priority"],
                "status": "REQUIRES HUMAN REVIEW",
            },
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
