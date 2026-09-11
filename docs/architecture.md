# Architecture

```text
Uploaded video -> OpenCV frame sampling
              -> vehicle detector (suppression)
              -> custom SAR detector + COCO fallback
              -> overlap fusion -> ByteTrack IDs
              -> candidate identity layer -> repeated observations
              -> pixel-to-GPS estimate -> triage score and GeoJSON
```

All candidate records are recommendations for human verification. The application does not dispatch responders or make an autonomous rescue decision.
