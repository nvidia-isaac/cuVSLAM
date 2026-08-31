# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
from numpy import loadtxt
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
import torch
import torchvision.io
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cuvslam

# Set up dataset path
sequence_path = os.path.join(
    os.path.dirname(__file__),
    "dataset/sequences/06"
)

# Load segmentation model to GPU
print("Loading segmentation model...")
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available for PyTorch. This example requires a system with a CUDA-capable NVIDIA GPU. "
        "Check your PyTorch installation with: python3 -c \"import torch; print(torch.cuda.is_available())\""
    )
device = torch.device("cuda")
print(f"Using device: {device}")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = seg_model.to(device)
print(f"Segmentation model loaded on {device}")

# Generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

# Setup rerun visualizer
rr.init('kitti', strict=True, spawn=True)  # launch re-run instance

# Setup rerun views
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(
        row_shares=[0.3, 0.2, 0.2],
        contents=[
            rrb.Spatial3DView(),
            rrb.Spatial2DView(origin='car/cam0'),
            rrb.Spatial2DView(origin='car/mask0')
        ]
    )
))

# Setup coordinate basis for root, cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Draw arrays in origin X-red, Y-green, Z-blue
rr.log("xyz", rr.Arrows3D(
    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=['[x]', '[y]', '[z]']
), static=True)

# Load KITTI dataset calibration and initilize cameras
intrinsics = loadtxt(
    os.path.join(sequence_path, 'calib.txt'),
    usecols=range(1, 13)
)[:4].reshape(4, 3, 4)

size = Image.open(os.path.join(sequence_path, 'image_0', '000001.png')).size

cameras = [cuvslam.Camera(), cuvslam.Camera()]
for i in [0, 1]:
    cameras[i].size = size
    cameras[i].principal = [intrinsics[i][0][2], intrinsics[i][1][2]]
    cameras[i].focal = [intrinsics[i].diagonal()[0], intrinsics[i].diagonal()[1]]
cameras[1].rig_from_camera.translation[0] = -intrinsics[1][0][3] / intrinsics[1][0][0]

# Initialize the cuvslam tracker
odom_cfg = cuvslam.Odometry.Config(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True
)
tracker = cuvslam.Tracker(cuvslam.Rig(cameras), odom_cfg)

timestamps = [
    int(10 ** 9 * float(sec_str))
    for sec_str in open(os.path.join(sequence_path, 'times.txt')).readlines()
]

# Track each frames in the dataset sequence
trajectory = []
prev_bin_mask_tensor = None  # Store previous binary mask tensor for temporal smoothing
for frame in range(len(timestamps)):
    # Load images directly as GPU tensors using torchvision
    image_paths = [
        os.path.join(sequence_path, f'image_{cam}', f'{frame:0>6}.png')
        for cam in [0, 1]
    ]
    images_gpu = []

    for path in image_paths:
        # Read image directly as tensor and move to GPU
        img_tensor = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.UNCHANGED).to(device)
        # Ensure uint8 format
        if img_tensor.dtype != torch.uint8:
            img_tensor = img_tensor.to(torch.uint8)
        # Reshape from (C, H, W) to (H, W, C) for track method
        img_tensor = img_tensor.permute(1, 2, 0)
        images_gpu.append(img_tensor)

    # For segmentation, we still need PIL format for the processor
    left_image_pil = Image.open(image_paths[0])
    if left_image_pil.mode != 'RGB':
        left_image_pil = left_image_pil.convert('RGB')

    # Process image for segmentation
    inputs = processor(images=left_image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = outputs.logits

    # Upsample logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=left_image_pil.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    # Get predicted segmentation map and create binary mask for cars (keep on GPU)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    car_class_id = 20  # Car class in ADE20K dataset
    bin_mask_tensor = (pred_seg == car_class_id).to(torch.uint8) * 255

    # Apply temporal smoothing: if pixel was non-zero in previous frame, keep it as 255
    if prev_bin_mask_tensor is not None:
        current_bin_mask_tensor = torch.where(prev_bin_mask_tensor > 0, 255, bin_mask_tensor)
    else:
        current_bin_mask_tensor = bin_mask_tensor

    # Store current mask for next iteration
    prev_bin_mask_tensor = bin_mask_tensor.clone()

    # Create mask tensors for both cameras
    masks_gpu = [current_bin_mask_tensor, torch.zeros_like(current_bin_mask_tensor, dtype=torch.uint8)]

    # Do stereo visual by PyCuVSLAM with GPU tensors
    odom_pose_estimate, _ = tracker.track(timestamps[frame], images_gpu, masks_gpu)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # Get visualization data
    observations = tracker.odometry.get_last_observations(0)  # get observation from left camera
    landmarks = tracker.odometry.get_last_landmarks()
    final_landmarks = tracker.odometry.get_final_landmarks()

    # Prepare visualization data
    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)

    # Send results to rerun for visualization (convert to CPU only here)
    rr.set_time('frame', sequence=frame)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.1))
    rr.log('car', rr.Transform3D(
        translation=odom_pose.translation,
        quaternion=odom_pose.rotation
    ))
    rr.log('car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
    rr.log('car/landmarks_center', rr.Points3D(
        landmark_xyz, radii=0.25, colors=landmarks_colors
    ))
    rr.log('car/landmarks_lines', rr.Arrows3D(
        vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
    ))
    rr.log('car/cam0', rr.Pinhole(
        image_plane_distance=1.68,
        image_from_camera=intrinsics[0][:3, :3],
        width=size[0],
        height=size[1]
    ))
    rr.log('car/cam0/image', rr.Image(
        images_gpu[0].permute(2, 0, 1).cpu().numpy()
    ).compress(jpeg_quality=80))
    rr.log('car/cam0/observations', rr.Points2D(
        observations_uv, radii=5, colors=observations_colors
    ))
    rr.log('car/mask0', rr.Pinhole(
        image_plane_distance=1.68,
        image_from_camera=intrinsics[0][:3, :3],
        width=size[0],
        height=size[1]
    ))
    rr.log('car/mask0/image', rr.Image(
        current_bin_mask_tensor.cpu().numpy()
    ).compress(jpeg_quality=80))
