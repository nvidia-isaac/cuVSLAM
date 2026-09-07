
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

#include <chrono>
#include <iostream>

#include "gflags/gflags.h"

#include "camera_rig_edex/blackout_oscillator_filter.h"
#include "camera_rig_edex/repeated_camera_rig_edex.h"
#include "camera_rig_edex/shuttle_camera_rig_edex.h"
#include "common/coordinate_system.h"
#include "common/environment.h"
#include "common/include_json.h"
#include "common/log_types.h"
#include "common/rerun.h"
#include "edex/edex.h"
#include "launcher/launcher_create.h"
#include "log/log.h"
#include "odometry/svo_config.h"
#include "params/cli.h"
#include "visualizer/visualizer.hpp"

DEFINE_string(logger_filename, "", "Log filename");
DEFINE_string(edex, "", "Path to folder with edex_file");
DEFINE_string(edex_filename, "stereo.edex", "Edex filename");
DEFINE_string(output_edex, "", "Path to output edex file. If empty, no output edex is stored.");
DEFINE_string(output_poses, "", "Path to output poses file (KITTI format). If empty, no poses are stored.");
DEFINE_string(gt_file, "", "Path to ground truth poses file (KITTI format). If empty, no GT comparison.");
DEFINE_int32(camera_id, 0, "camera_id");
DEFINE_bool(verbose, true, "Enable verbosity");
DEFINE_bool(use_slam, false, "use slam");
DEFINE_bool(filter_tracks, true, "Filter 2D tracks");
DEFINE_bool(use_seq_path, true, "Search for edex file inside CUVSLAM_DATASETS path");
DEFINE_int32(sequence_num_repeats, 1, "How many times repeat the sequence");
DEFINE_string(repeat_type, "", "Type of repeating the sequence: '', 'Repeat' or 'Shuttle' ");
DEFINE_int32(blackout_oscillator, 0, "Do blackout each [blackout_oscillator] frame");
DEFINE_int32(blackout_oscillator_duration, 10, "Number of blackouted frames in each series");

using namespace cuvslam;
using namespace Json;
using namespace JsonUtils;

DEFINE_string(sba_mode, "gpu",
              "Bundle adjuster: none|cpu|gpu|imu|imugpu. Chosen when the tracker is built, so it is a\n"
              "\tflag rather than a -P parameter");
DEFINE_bool(list_params, false, "Print every tunable parameter with its type, default and description, then exit");

int main(int argC, char** ppArgV) {
  std::cout << "Welcome to nVidia cuVSLAM tracker.\n\n";
  Trace::SetVerbosity(Trace::Verbosity::Debug);
  const std::vector<std::string> param_overrides = cuvslam::params::ExtractOverrides(argC, ppArgV);
  std::vector<char*> all_args(ppArgV, ppArgV + argC);
  const std::string usage{"Usage: -edex <edex_folder> -edex_filename <edex_file> [-Pkey=value ...]\n"};
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argC, &ppArgV, /*remove flags = */ true);
  if (argC != 1) {
    std::cout << "This tools doesn't expect command line arguments, only flags listed with --help" << std::endl
              << "If you see this message, please check your command line." << std::endl;
    return EXIT_FAILURE;
  }

  odom::Settings svo_settings;
  cuvslam::params::Registry params;
  params.Add("sof", svo_settings.sof_settings);
  params.Add("sof.feature_selection", svo_settings.sof_settings.feature_selection_settings);
  params.Add("kf", svo_settings.kf_settings);
  params.Add("sba", svo_settings.sba_settings);
  params.Add("sm", svo_settings.sm_settings);

  if (FLAGS_list_params) {
    cuvslam::params::PrintReference(params, std::cout);
    return EXIT_SUCCESS;
  }
  try {
    svo_settings.sba_settings.mode = cuvslam::params::ParseEnum<sba::Mode>(FLAGS_sba_mode);
    cuvslam::params::ApplyOverrides(params, param_overrides, cuvslam::params::Source::CommandLine);
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // check environment
  std::string sequence_folder = Environment::GetVar(Environment::CUVSLAM_DATASETS);
  if (!IsPathEndWithSlash(sequence_folder)) sequence_folder += '/';

  if (sequence_folder.empty()) {
    std::cout << "Can't understand where is sequence folder. Check CUVSLAM_DATASETS environment variable." << std::endl;
    return 0;
  }

  if (FLAGS_edex.empty()) {
    std::cout << "Please provide path to folder with edex file with --edex option" << std::endl;
    return 0;
  }

  std::string edex_dir = FLAGS_edex;
  if (!IsPathEndWithSlash(edex_dir)) {
    edex_dir += '/';
  }
  if (FLAGS_use_seq_path) {
    edex_dir = sequence_folder + edex_dir;
  }
  const std::string edex_filename = FLAGS_edex_filename;
  const std::string out_edex_filename = "out_" + edex_filename;
  std::string full_filename = edex_dir + edex_filename;

  if (!FLAGS_logger_filename.empty() && FLAGS_use_seq_path)
    FLAGS_logger_filename = sequence_folder + FLAGS_logger_filename;

  if (!FLAGS_logger_filename.empty()) {
#ifdef CUVSLAM_LOG_ENABLE
    std::cout << "Create spd logger: " << FLAGS_logger_filename << std::endl;
    auto logger = log::CreateSpdlogLogger(FLAGS_logger_filename.c_str());
    log::SetLogger(logger);
#else  // !CUVSLAM_LOG_ENABLE
    std::cout << "No CUVSLAM_LOG_ENABLE definition. Flag -logger_filename will be ignored " << std::endl;
#endif
    Json::Value root;

    root["logs"]["LogFrames"] = (LogFrames::enable_ ? 1 : 0);
    root["logs"]["LogTracks2d"] = (LogTracks2d::enable_ ? 1 : 0);
    root["logs"]["LogTracks3d"] = (LogTracks3d::enable_ ? 1 : 0);
    root["logs"]["LogSba"] = (LogSba::enable_ ? 1 : 0);

    int args_count = static_cast<int>(all_args.size());
    for (int i = 0; i < args_count; i++) {
      root["args"][i] = all_args[i];
    }
    log::Json<LogRoot>(root);
  }

  ErrorCode err;

  std::cout << "Start tracking " << full_filename << " with camera " << FLAGS_camera_id << std::endl;

  edex::EdexFile f;
  if (!f.read(full_filename)) {
    std::cout << "Can't read edex file" << std::endl;
    return -1;
  }

  std::unique_ptr<ICameraRig> rig;
  if (FLAGS_repeat_type == "Shuttle") {
    rig = std::make_unique<camera_rig_edex::ShuttleCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(full_filename), FLAGS_sequence_num_repeats);
  } else if (FLAGS_repeat_type == "Repeat") {
    rig = std::make_unique<camera_rig_edex::RepeatedCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(full_filename), FLAGS_sequence_num_repeats);
  } else {
    rig = std::make_unique<camera_rig_edex::CameraRigEdex>(full_filename);
  }

  // black oscillator
  if (FLAGS_blackout_oscillator != 0) {
    std::cout << "Assign Blackout Oscilator Filter. Period=" << FLAGS_blackout_oscillator << ", duration "
              << FLAGS_blackout_oscillator_duration << std::endl;
    std::unique_ptr<ICameraRig> filter = std::make_unique<camera_rig_edex::BlackoutOscillatorFilter>(
        std::move(rig), FLAGS_blackout_oscillator, FLAGS_blackout_oscillator_duration);
    rig = std::move(filter);
  }

  const Isometry3T rig_from_imu = LegacyEdexImuExtrinsicToOpenCV(f.imu_.transform);
  float gyroscope_noise_density = f.imu_.gyroscope_noise_density;          // rad / (s * srqt(Hz))
  float gyroscope_random_walk = f.imu_.gyroscope_random_walk;              // rad / (s ^ 2 * srqt(Hz))
  float accelerometer_noise_density = f.imu_.accelerometer_noise_density;  // m / (s ^ 2 * srqt(Hz))
  float accelerometer_random_walk = f.imu_.accelerometer_random_walk;      // m / (s ^ 3 * srqt(Hz))
  float frequency = f.imu_.frequency;                                      // Hz

  svo_settings.imu_calibration = imu::ImuCalibration(rig_from_imu, gyroscope_noise_density, gyroscope_random_walk,
                                                     accelerometer_noise_density, accelerometer_random_walk, frequency);
  svo_settings.verbose = FLAGS_verbose;

  std::unique_ptr<launcher::BaseLauncher> launcher = launcher::CreateLauncher(*rig, svo_settings);

  if (!FLAGS_gt_file.empty()) {
    Isometry3TVector poses_gt;
    if (edex::loadPoses(FLAGS_gt_file, poses_gt)) {
      launcher->updateGTPoses(poses_gt);
    } else {
      std::cout << "Warning: Could not load ground truth poses from " << FLAGS_gt_file << std::endl;
    }
  }

  if (FLAGS_use_slam) {
    launcher->SetupSlam();
  }

  const auto start = std::chrono::steady_clock::now();
  if (!(err = launcher->launch())) {
    std::cout << "Track error: " << ErrorCode::GetString(err) << std::endl;
    RERUN(cuvslam::shutdownVisualizer);
    return 0;
  }

  const auto end = std::chrono::steady_clock::now();
  const std::chrono::duration<float> elapsed_seconds = end - start;
  const float seconds = elapsed_seconds.count();
  assert(seconds != 0.f);
  const float fps = launcher->nFrames() / seconds;

  // analyse result
  const float batch_resdual = launcher->calcBatchResidual();
  size_t minVisible2DTracks;
  float averageOnlineResidual, maxFrameResidual, averageVisible2DTracks;
  FrameId maxFrameResidualFrame, minVisible2DTracksFrame;
  launcher->calcOnlineResidual(averageOnlineResidual, maxFrameResidual, maxFrameResidualFrame);
  launcher->calcVisible2DTracksStats(averageVisible2DTracks, minVisible2DTracks, minVisible2DTracksFrame);
  float average_track_sec, max_track_sec;
  FrameId max_track_frame;
  launcher->getTimers(average_track_sec, max_track_sec, max_track_frame);
  bool max_track_highlite = max_track_sec > 1 / 60.;

  log::Value<LogRoot>("batch_residual", batch_resdual);
  log::Value<LogRoot>("average_online_residual", averageOnlineResidual);
  log::Value<LogRoot>("max_frame_residual", maxFrameResidual);
  log::Value<LogRoot>("max_frame_residual_frame", static_cast<size_t>(maxFrameResidualFrame));

  log::Value<LogRoot>("average_visible_2dtracks", averageVisible2DTracks);
  log::Value<LogRoot>("min_visible_2dtracks", minVisible2DTracks);
  log::Value<LogRoot>("min_visible_2dtracks_frame", static_cast<size_t>(minVisible2DTracksFrame));

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "batch residual = " << batch_resdual;
  std::cout << std::endl;
  std::cout << "average online residual = " << averageOnlineResidual;
  std::cout << std::endl;
  std::cout << "max frame residual = " << maxFrameResidual << " occurred in " << maxFrameResidualFrame << " frame";
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "average visible observations number = " << averageVisible2DTracks;
  std::cout << std::endl;
  std::cout << "min visible observations number = " << minVisible2DTracks << " occured in " << minVisible2DTracksFrame
            << " frame";
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "fps = " << fps << ", duration = " << seconds << " seconds" << std::endl;
  std::cout << "average track() duration = " << average_track_sec << "s, max track() duration = ";

  if (max_track_highlite) {
    std::cout << "\033[31m";
  }
  std::cout << max_track_sec << "s (in frame #" << max_track_frame << ")" << std::endl;
  if (max_track_highlite) {
    std::cout << "\033[0m";
  }
  std::cout << std::endl;

  {
    log::Scoped<LogRoot> stat_sc("statistic");

    std::vector<float> value = {seconds, (float)launcher->nFrames()};
    log::Value<LogRoot>("full_frame_stopwatch", value.begin(), value.end());
  }

  // write poses to KITTI format file
  const std::string poses_file = FLAGS_output_poses;
  if (!poses_file.empty()) {
    const auto& camera_map = launcher->cameraMap();
    Isometry3TVector poses_result;
    poses_result.reserve(camera_map.size());
    for (const auto& [frame_id, pose] : camera_map) {
      poses_result.push_back(pose);
    }
    if (!edex::writePoses(poses_file, poses_result)) {
      std::cout << "Can't write poses to " << poses_file << std::endl;
      RERUN(cuvslam::shutdownVisualizer);
      return 0;
    }
  }

  // save result
  const std::string output_edex = FLAGS_output_edex;
  if (output_edex.empty()) {
    std::cout << "Tip: You can set the `-output_edex` argument to store the result." << std::endl;
    RERUN(cuvslam::shutdownVisualizer);
    return 0;
  }

  std::cout << "Start saving result to " << output_edex << std::endl;

  if (!launcher->saveResultToEdex(output_edex, rig->getIntrinsic(0), FLAGS_filter_tracks,
                                  edex::RotationStyle::RotationMatrix)) {
    std::cout << "Can't save result." << std::endl;
    RERUN(cuvslam::shutdownVisualizer);
    return 0;
  }

  RERUN(cuvslam::shutdownVisualizer);

  return 0;
}
