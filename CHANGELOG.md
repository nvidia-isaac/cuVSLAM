# Changelog

## [15.0.1] - 2026-09-17

### Security

- Updated auxiliary Python dependencies to Pillow 12.3.0 (CVE-2026-25990, CVE-2026-42311,
  CVE-2026-54058, CVE-2026-59197), NumPy 2.2.6 (CVE-2021-34141, CVE-2021-41495,
  CVE-2021-41496), SciPy 1.15.3 (CVE-2023-25399, CVE-2023-29824), fonttools 4.60.2
  (CVE-2025-66034), and WeasyPrint 69.0 (CVE-2025-68616, CVE-2026-49452)

## [15.0.0] - 2026-03-02

Initial open-source release.

### Added
- Cached map-to-disk SLAM mode
- RGBD pipeline optimization
- Rerun visualization of internal cuVSLAM data
- Examples moved into the repository (from a previously separate pycuvslam repo)
- Troubleshooting guide (`TROUBLESHOOTING.md`)

### Changed
- New NVIDIA Community License
- Switched to semantic versioning (from this release onward)
- SLAM internal refactoring is in progress
- Removed JPEG/PNG dependencies from libcuvslam
- Opened cuVSLAM API to accept user-provided `cudaStream_t`
- CMake 4+ build compatibility

### Removed
- C API (superseded by C++ API)
- Several internal methods and unused parameters from public SLAM API

### Fixed
- Memory leaks
- RGBD depth mask bug
- Multiple other bugfixes

### Security
- Updated libpng version
