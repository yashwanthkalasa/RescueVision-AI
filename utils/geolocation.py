"""Approximate nadir-camera pixel-to-GPS conversion for triage only."""

from __future__ import annotations

import math


def pixel_to_gps(bbox, frame_width, frame_height, drone_lat, drone_lon, altitude, yaw_deg, hfov=78.0, vfov=50.0):
    x1, _, x2, y2 = bbox
    pixel_x = (x1 + x2) / 2.0
    pixel_y = y2
    ground_width = 2 * altitude * math.tan(math.radians(hfov / 2))
    ground_height = 2 * altitude * math.tan(math.radians(vfov / 2))
    east = ((pixel_x - frame_width / 2) / frame_width) * ground_width
    north = -((pixel_y - frame_height / 2) / frame_height) * ground_height
    yaw = math.radians(yaw_deg)
    rotated_east = east * math.cos(yaw) - north * math.sin(yaw)
    rotated_north = east * math.sin(yaw) + north * math.cos(yaw)
    latitude = drone_lat + rotated_north / 111_320
    longitude = drone_lon + rotated_east / (111_320 * math.cos(math.radians(drone_lat)))
    return latitude, longitude
