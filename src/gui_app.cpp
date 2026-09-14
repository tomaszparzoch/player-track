#include "gui_app.hpp"

#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <opencv2/videoio.hpp>

#include "multi_camera_session.hpp"
#include "pitch_calibration.hpp"
#include "project_config.hpp"
#include "app_options.hpp"
#include "time_format.hpp"

#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace {

#ifdef _WIN32
std::string open_video_dialog() {
  char filename[MAX_PATH] = "";
  OPENFILENAMEA ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;
  ofn.lpstrFilter = "Video Files\0*.mp4;*.avi;*.mkv\0All Files\0*.*\0";
  ofn.lpstrFile = filename;
  ofn.nMaxFile = MAX_PATH;
  ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
  if (GetOpenFileNameA(&ofn)) return filename;
  return {};
}
#else
std::string open_video_dialog() { return {}; }
#endif

const char* role_labels[] = {"corner_tl", "corner_tr", "corner_bl", "corner_br",
                             "side_left", "side_right", "tactical_pan", "unknown"};

}  // namespace

int run_gui_app() {
  if (!glfwInit()) return 1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(1280, 800, "Player Track", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  ProjectConfig project;
  char output_dir[256];
  std::strncpy(output_dir, project.output_dir.c_str(), sizeof(output_dir) - 1);

  bool analyzing = false;
  double progress = 0.0;
  double elapsed_s = 0.0;
  double eta_s = -1.0;
  double last_analysis_duration_s = 0.0;
  std::string status = "Ready";
  MultiCameraResult last_result;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Player Track Project");
    ImGui::Text("Mode: %s", project_mode_to_string(project.mode()).c_str());
    ImGui::InputText("Output dir", output_dir, sizeof(output_dir));
    project.output_dir = output_dir;

    if (ImGui::Button("Add camera")) {
      const std::string path = open_video_dialog();
      if (!path.empty()) {
        CameraConfig cam;
        cam.video_path = path;
        cam.name = std::filesystem::path(path).stem().string();
        cam.motion = CameraMotion::Static;
        cam.world_coords = false;
        project.cameras.push_back(cam);
      }
    }

    for (int i = 0; i < static_cast<int>(project.cameras.size()); ++i) {
      auto& cam = project.cameras[i];
      ImGui::PushID(i);
      ImGui::Separator();
      ImGui::Text("Camera: %s", cam.name.c_str());
      ImGui::Text("File: %s", cam.video_path.c_str());
      ImGui::SliderFloat("Offset (s)", reinterpret_cast<float*>(&cam.offset_s), -30.0f, 30.0f);

      int role_idx = static_cast<int>(cam.role);
      if (ImGui::Combo("Role", &role_idx, role_labels, IM_ARRAYSIZE(role_labels))) {
        cam.role = static_cast<CameraRole>(role_idx);
        if (cam.role == CameraRole::TacticalPan) cam.motion = CameraMotion::Panning;
      }

      int motion_idx = cam.motion == CameraMotion::Panning ? 1 : 0;
      if (ImGui::Combo("Motion", &motion_idx, "static\0panning\0")) {
        cam.motion = motion_idx == 1 ? CameraMotion::Panning : CameraMotion::Static;
      }

      if (project.mode() != ProjectMode::TacticalOnly) {
        ImGui::Checkbox("Use for world coords", &cam.world_coords);
        if (!cam.world_coords) {
          ImGui::TextColored({0.7f, 0.9f, 0.7f, 1},
                             "Pixel mode: tracks/heatmaps in image coords (no calibration).");
        }
      } else {
        cam.world_coords = true;
        ImGui::Text("World coords: enabled (tactical only mode)");
      }

      if (cam.world_coords && !cam.is_panning()) {
        if (ImGui::Button("Calibrate")) {
          cv::VideoCapture cap(cam.video_path);
          cv::Mat frame;
          if (cap.read(frame) && !frame.empty()) {
            PitchCalibration calib;
            const PitchModel model(project.pitch_length_m, project.pitch_width_m);
            if (calib.calibrate_lines_interactive(frame, model)) {
              cam.calib_path = (std::filesystem::path(output_dir) / (cam.name + ".calib")).string();
              std::filesystem::create_directories(output_dir);
              calib.save(cam.calib_path);
              status = "Calibrated " + cam.name;
            }
          }
        }
        ImGui::SameLine();
      }
      if (ImGui::Button("Remove")) {
        project.cameras.erase(project.cameras.begin() + i);
        ImGui::PopID();
        break;
      }
      ImGui::PopID();
    }

    if (project.mode() == ProjectMode::TacticalOnly) {
      ImGui::TextColored({1, 0.7f, 0.2f, 1}, "Tactical-only mode: lower metric accuracy expected.");
    }

    ImGui::Separator();
    ImGui::Text("Performance");
    ImGui::SliderInt("Detect every N frames", &project.detect_frame_step, 1, 10);
    ImGui::SliderInt("Max detect width (px)", &project.detect_max_width, 640, 1920);
    ImGui::SliderInt("Detect input size", &project.detect_input_size, 320, 1280);
    ImGui::SliderInt("OCR every N detect frames", &project.ocr_frame_step, 5, 60);
    ImGui::Checkbox("Write annotated MP4", &project.write_annotated_video);

    int backend_idx = 0;
    switch (project.dnn_backend) {
      case DnnBackend::Cpu: backend_idx = 1; break;
      case DnnBackend::Cuda: backend_idx = 2; break;
      case DnnBackend::DirectML: backend_idx = 3; break;
      case DnnBackend::OpenVino: backend_idx = 4; break;
      default: backend_idx = 0; break;
    }
    if (ImGui::Combo("Inference provider", &backend_idx,
                     "auto (GPU on Windows)\0cpu\0cuda\0directml\0openvino\0")) {
      switch (backend_idx) {
        case 1: project.dnn_backend = DnnBackend::Cpu; break;
        case 2: project.dnn_backend = DnnBackend::Cuda; break;
        case 3: project.dnn_backend = DnnBackend::DirectML; break;
        case 4: project.dnn_backend = DnnBackend::OpenVino; break;
        default: project.dnn_backend = DnnBackend::Auto; break;
      }
    }
    ImGui::TextWrapped(
        "GPU inference uses ONNX Runtime + DirectML on Windows (works without CUDA toolkit). "
        "Speed tips: detect every 2-3 frames, max width 960, input 416, yolov8n, disable MP4.");

    ImGui::Separator();
    if (!analyzing && ImGui::Button("Run analysis") && !project.cameras.empty()) {
      analyzing = true;
      progress = 0.0;
      elapsed_s = 0.0;
      eta_s = -1.0;
      status = "Analyzing...";
      const auto analysis_start = std::chrono::steady_clock::now();
      std::thread([&, project, analysis_start]() {
        last_result = run_multi_camera_session(project, [&](double p, const std::string& msg) {
          progress = p;
          status = msg;
          elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - analysis_start)
                          .count();
          eta_s = estimate_remaining_seconds(p, elapsed_s);
        });
        analyzing = false;
        last_analysis_duration_s = last_result.analysis_duration_s;
        if (last_result.success) {
          status = "Analysis complete (" + format_duration(last_analysis_duration_s) + ")";
        } else {
          status = last_result.error;
        }
      }).detach();
    }

    if (analyzing) {
      ImGui::ProgressBar(static_cast<float>(progress));
      ImGui::Text("%s", status.c_str());
      ImGui::Text("Elapsed: %s", format_duration(elapsed_s).c_str());
      if (eta_s >= 0.0) {
        ImGui::Text("ETA: ~%s", format_duration(eta_s).c_str());
      } else {
        ImGui::Text("ETA: estimating...");
      }
    } else {
      ImGui::Text("%s", status.c_str());
      if (last_analysis_duration_s > 0.0 && last_result.success) {
        ImGui::Text("Analysis time: %s", format_duration(last_analysis_duration_s).c_str());
      }
      if (last_result.success) {
        ImGui::Text("PDF: %s", last_result.pdf_report.string().c_str());
        ImGui::Text("CSV: %s", last_result.global_csv.string().c_str());
        for (const auto& v : last_result.annotated_videos) {
          ImGui::Text("Video: %s", v.string().c_str());
        }
      }
    }

    if (ImGui::Button("Save project")) {
      save_project_config("project.ptproj", project);
      status = "Saved project.ptproj";
    }
    ImGui::SameLine();
    if (ImGui::Button("Load project")) {
      if (auto loaded = load_project_config("project.ptproj")) {
        project = *loaded;
        std::strncpy(output_dir, project.output_dir.c_str(), sizeof(output_dir) - 1);
        status = "Loaded project.ptproj";
      }
    }

    ImGui::End();
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
