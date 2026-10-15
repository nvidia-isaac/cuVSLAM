# Public datasets for evaluating place recognition

`cuvslam_vpr_reporter` scores a backend on **two traversals of the same route**. It needs, from each traversal,
per-frame images and a 6-DoF ground truth pose, and it needs both traversals' ground truth to be in **one common
frame** — that is what lets a match be scored by the distance between the matched map frame and the query frame.

That requirement is what separates the two kinds of dataset below. The classic VPR benchmarks (Pittsburgh,
Tokyo 24/7, Nordland) are image-retrieval sets: they give you a database, a query set and a correctness label, but
no trajectory and no metric poses, so they measure recall and nothing else. The SLAM-style datasets give poses, so
they also measure how far wrong a wrong answer was — which is the number a robot actually cares about.

## Usable by this tool

| Dataset | Stresses | Scale | Sensors / ground truth | Access |
|---------|----------|-------|------------------------|--------|
| [4Seasons](https://cvg.cit.tum.de/data/datasets/4seasons-dataset) | season, weather, day/night, tunnels, garages | >350 km over 9 environments | stereo + IMU; cm-accurate globally consistent poses from stereo VIO fused with RTK-GNSS | free, registration |
| [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) | weather, illumination, season, traffic | 100+ traversals of one 10 km route over a year | Bumblebee XB3 stereo, LIDAR, GPS/INS | free, registration, non-commercial |
| [RobotCar Seasons](https://www.visuallocalization.net/datasets/) | the above, curated for relocalization | 10 conditions | reference 6-DoF poses per condition | free |
| [NCLT](https://robots.engin.umich.edu/nclt/) | indoor **and** outdoor, 15 months, varying routes | 27 sessions, 147.4 km, 34.9 h | omnidirectional camera, LIDAR, GT poses | free |
| [Boreas](https://www.boreas.utias.utoronto.ca/) | season, rain, falling snow | >350 km, one route repeated over a year | 5 MP camera, 128-beam LIDAR, radar; cm-accurate post-processed poses | free |
| [Extended CMU Seasons](https://www.visuallocalization.net/datasets/) | season, vegetation, suburban and park | 12 traversals, ~9 km | 2 cameras, reference 6-DoF poses | free |
| [CODa](https://amrl.cs.utexas.edu/coda/) | repeated campus traversals, pedestrians | 23 sequences | rectified stereo, LiDAR-derived poses | license-gated (TX Dataverse) |
| [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | indoor, MAV viewpoint; MH_01–05 revisit one hall | 11 sequences | stereo + IMU, Vicon/Leica | free |
| [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | within-sequence loop closure | 22 sequences | stereo, GPS/INS | free |
| [M3ED](https://m3ed.io/) | legged and aggressive motion | 19 SPOT sequences | stereo, event cameras, FasterLIO poses | free |

`prepare_coda`, `prepare_euroc`, `prepare_kitti`, `prepare_m3ed_spot` and `prepare_tartan` already convert the
last five into the EDEX layout the reporter reads. The first five need a converter written for them.

## Retrieval-only benchmarks

Standard in the VPR literature, and the right choice for comparing a *descriptor* against published numbers — but
they cannot drive this tool, because there is no trajectory to replay and no metric pose to be wrong by.

| Dataset | Stresses | Scale | Why it does not fit |
|---------|----------|-------|---------------------|
| [Nordland](https://webdiis.unizar.es/~jmfacil/pr-nordland/) | extreme season change with **zero** viewpoint change | 4 traversals of a 728 km rail route | frame-index alignment only; no 6-DoF ground truth |
| [Pittsburgh 250k / Pitts30k](https://www.di.ens.fr/willow/research/netvlad/) | urban viewpoint change | 250 k database, 24 k queries | Street View panorama crops, not a continuous traversal |
| [Tokyo 24/7](https://www.ok.sc.e.titech.ac.jp/INDEX/) | day → sunset → night | 76 k database, 315 queries | query set, not a traversal |
| [Mapillary SLS](https://www.mapillary.com/dataset/places) | 30 cities, crowd-sourced, illumination and weather | 1.6 M images | sequences exist but ground truth is GPS only |
| [Gardens Point Walking](https://zenodo.org/record/4590133) | lateral viewpoint plus day/night | 3 × 200 images | no poses — but a useful 30-second sanity check |

## What to add next

1. **4Seasons** — cm-accurate poses across seasons in a single global frame is exactly what the tool's metric
   assumes, and it is the only free dataset that gives it.
2. **Oxford RobotCar / RobotCar Seasons** — the field's reference long-term benchmark, so numbers from it are
   comparable to published work.
3. **NCLT** — the only one of these that crosses the indoor/outdoor boundary, where a thumbnail or a bag of words
   behaves very differently from a learned descriptor.
