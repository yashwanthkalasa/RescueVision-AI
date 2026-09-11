# Methodology and Limitations

The MVP uses a custom aerial human detector when available, a COCO `person` fallback, and a YOLO vehicle detector to suppress candidate boxes overlapping cars, motorcycles, buses, or trucks. Duplicate model detections use overlap-over-smaller-box at `0.60`. Vehicle suppression uses candidate-box overlap at `0.50`, deliberately requiring substantial containment rather than proximity.

ByteTrack is invoked by Ultralytics with `bytetrack.yaml`. A candidate is promoted after three observations. If a tracker ID changes, the identity layer associates the new track using recent box overlap or image-space center proximity, never GPS distance alone.

GPS is an approximate nadir-camera projection using the bottom-center of the box, a flat-ground assumption, manually entered telemetry, and 78° x 50° field of view. It does not correct roll, pitch, lens distortion, terrain elevation, or time-varying drone pose, and is not survey-grade.
