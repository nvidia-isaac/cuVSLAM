"""Producing and describing a recorded SLAM session: readers, trajectories, mapping.

These are the only modules that know about EuRoC on disk or drive the native tracker.
The repair policies consume the JSON these produce, never the dataset itself.
"""
