
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#warning GravityCalc should be fixed
#if 0
#include "gflags/gflags.h"

#include <iostream>

#include "camera_rig_edex/camera_rig_edex.h"
#include "common/environment.h"
#include "common/imu_measurement.h"
#include "imu/imu_sba_problem.h"
#include "imu/inertial_optimization.h"
#include "odometry/svo_config.h"


DEFINE_string(sequence, "euroc/V2_03_difficult", "Sequence name");

DEFINE_int32(end_frame, -1, "End frame");
DEFINE_double(gnd, 0.00016968, "Gyroscope noise density");
DEFINE_double(grw, 0.000019393, "Gyroscope random walk");
DEFINE_double(and, 0.002, "Accelerometer noise density");
DEFINE_double(arw, 0.003, "Accelerometer random walk");
DEFINE_double(freq, 200, "IMU frequency");

using namespace cuvslam;

bool loadPoses(const std::string& file_name, Isometry3TVector& poses)
{
    FILE* fp = std::fopen(file_name.c_str(), "r");

    if (!fp)
    {
        return false;
    }

    while (!std::feof(fp))
    {
        Isometry3T pose;
        Matrix4T& m = pose.matrix();
        float fm[12];

        for (size_t k = 0; k < 12; ++k)
        {
            if (!std::fscanf(fp, "%f", &fm[k]))
            {
                assert(false);
                std::fclose(fp);
                return false;
            }
        }

        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                m(i, j) = fm[i * 4 + j];
            }
        }

        assert(m.allFinite());
        pose.makeAffine();
        poses.push_back(pose);
    }

    std::fclose(fp);

    poses.pop_back(); // current function somehow adds the last pose twice, get rid of it

    if (!poses[0].isApprox(Isometry3T::Identity(), 1e-3)) {
        Isometry3T start_pose_inverse = poses[0].inverse();
        for (Isometry3T& p: poses) {
            p = start_pose_inverse * p;
        }
    }
    return true;
}

int main(int argC, char** ppArgV) {
    std::vector<char *> all_args(ppArgV, ppArgV + argC);
    gflags::ParseCommandLineFlags(&argC, &ppArgV, true);
    std::string cfgFolder = Environment::GetVar(Environment::CUVSLAM_DATASETS);

    if (!IsPathEndWithSlash(cfgFolder)) {
        cfgFolder += "/";
    }
    std::string sequence_path = cfgFolder + FLAGS_sequence;
    if (!IsPathEndWithSlash(sequence_path)) {
        sequence_path += "/";
    }
    const std::string edex_path = sequence_path + "stereo.edex";
    const std::string gt_path = sequence_path + "gt.txt";

    std::vector<Isometry3T> gt_poses;
    if (!loadPoses(gt_path, gt_poses)) {
        TraceError("Cant load poses");
        std::cout << "gt_path = " << gt_path << std::endl;
    }

    camera_rig_edex::CameraRigEdex rig(edex_path);
    if (rig.start() != ErrorCode::S_True) {
        TraceError("Failed to start");
    }

    edex::EdexFile f;
    if (!f.read(edex_path)) {
        std::cout << "Can't read edex file" << std::endl;
        return -1;
    }

    const imu::ImuCalibration calib {
        f.m_imu.transform,
        static_cast<float>(FLAGS_gnd),
        static_cast<float>(FLAGS_grw),
        static_cast<float>(FLAGS_and),
        static_cast<float>(FLAGS_arw),
        static_cast<float>(FLAGS_freq)
    };


    std::vector<imu::ImuMeasurement> measurements;
    imu::ImuMeasurementStorage storage(1e5);
    rig.registerIMUCallback([&](const imu::ImuMeasurement& m){
        measurements.push_back(m);
        storage.push_back(m);
    });

    Isometry3T cuVSLAMFromKitti;
    cuVSLAMFromKitti.linear() << 1, 0, 0,
                                0, -1, 0,
                                0, 0, -1;
    cuVSLAMFromKitti.translation().setZero();
    cuVSLAMFromKitti.makeAffine();

    std::vector<ImageSource> curr_sources;
    std::vector<ImageSource> masks_sources;
    std::vector<ImageMeta> curr_meta;

    std::vector<sba_imu::Pose> poses;

    int pose_id = 0;
    while (true) {
        if (FLAGS_end_frame > 0) {
            if (pose_id >= FLAGS_end_frame) {
                break;
            }
        }
        if (rig.getFrame(curr_sources, curr_meta, masks_sources) != ErrorCode::S_True) {
            break;
        }
        sba_imu::Pose p;
        {
            sba_imu::IMUPreintegration preint;
            for (const auto& m: measurements) {
                preint.IntegrateNewMeasurement(calib, m);
            }
            measurements.clear();

            p.w_from_imu = cuVSLAMFromKitti * gt_poses[pose_id] * cuVSLAMFromKitti.inverse() * calib.rig_from_imu();
            p.velocity.setZero();
            p.gyro_bias.setZero();
            p.acc_bias.setZero();

            if (!poses.empty()) {
                poses.back().preintegration = preint;
            }
        }
        poses.emplace_back(p);
        pose_id++;
    }

    std::cout << " num poses = " << poses.size() << std::endl;

    sba_imu::InertialOptimizer optimizer(storage);

    Matrix3T R;
    optimizer.optimize_inertial(calib, poses, R);

    std::cout << "gravity_w = " << std::endl << R * optimizer.get_default_gravity() << std::endl;

    return 0;
}

#else
int main() { return 0; }
#endif
