API Reference
=============

.. module:: cuvslam

Tracker coordinates frame and IMU processing across Odometry and optional Slam. Odometry estimates
rig motion from sensor input; Slam builds and optimizes a reusable map from odometry results.

Main classes
------------

.. autoclass:: Tracker
   :members:

.. autoclass:: Odometry
   :members:

.. autoclass:: Slam
   :members:

Data Structures
---------------

.. autoclass:: Pose
   :members:

.. autoclass:: Distortion
   :members:

.. autoclass:: Camera
   :members:

.. autoclass:: ImuCalibration
   :members:

.. autoclass:: Rig
   :members:

.. autoclass:: PoseStamped
   :members:

.. autoclass:: PoseWithCovariance
   :members:

.. autoclass:: PoseEstimate
   :members:

.. autoclass:: ImuMeasurement
   :members:

.. autoclass:: Landmark
   :members:

.. autoclass:: Observation
   :members:

Functions
---------

.. autofunction:: get_version

.. autofunction:: set_verbosity

.. autofunction:: warm_up_gpu
