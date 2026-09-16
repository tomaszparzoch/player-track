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
#include "pitch_model.hpp"
#include "project_config.hpp"
#include "camera_sidecar.hpp"
#include "app_options.hpp"
#include "path_utils.hpp"
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

const char* role_labels[] = {
    "Naroznik gora-lewo",
    "Naroznik gora-prawo",
    "Naroznik dol-lewo",
    "Naroznik dol-prawo",
    "Linia boczna (lewa)",
    "Linia boczna (prawa)",
    "Kamera taktyczna (pan)",
    "Nieznana / inna",
};

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
  output_dir[sizeof(output_dir) - 1] = '\0';

  const std::filesystem::path session_file = last_session_config_path();
  std::filesystem::path project_file = session_file;
  if (const auto found = resolve_resource_path("project.ptproj");
      std::filesystem::exists(found) && found.is_absolute()) {
    project_file = found;
  } else {
    project_file = application_directory() / "project.ptproj";
  }

  auto persist_config = [&](const ProjectConfig& config) {
    save_project_config(session_file, config);
    save_project_config(project_file, config);
  };

  bool analyzing = false;
  bool item_was_active = false;
  double progress = 0.0;
  double elapsed_s = 0.0;
  double eta_s = -1.0;
  double last_analysis_duration_s = 0.0;
  std::string status = "Ready";
  MultiCameraResult last_result;

  if (std::filesystem::exists(session_file)) {
    if (auto loaded = load_project_config(session_file)) {
      project = *loaded;
      std::strncpy(output_dir, project.output_dir.c_str(), sizeof(output_dir) - 1);
      output_dir[sizeof(output_dir) - 1] = '\0';
      status = "Wczytano ostatnia sesje";
    }
  } else if (std::filesystem::exists(project_file)) {
    if (auto loaded = load_project_config(project_file)) {
      project = *loaded;
      std::strncpy(output_dir, project.output_dir.c_str(), sizeof(output_dir) - 1);
      output_dir[sizeof(output_dir) - 1] = '\0';
      persist_config(project);
      status = "Loaded " + project_file.filename().string();
    }
  }

  auto apply_output_dir = [&]() {
    output_dir[sizeof(output_dir) - 1] = '\0';
    project.output_dir = output_dir;
  };

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Player Track Project");
    ImGui::Text("Mode: %s", project_mode_to_string(project.mode()).c_str());
    ImGui::InputText("Output dir", output_dir, sizeof(output_dir));
    apply_output_dir();

    ImGui::Separator();
    ImGui::Text("Boisko");
    ImGui::InputDouble("Dlugosc (m)", &project.pitch_length_m, 0.5, 1.0, "%.1f");
    ImGui::InputDouble("Szerokosc (m)", &project.pitch_width_m, 0.5, 1.0, "%.1f");
    if (project.pitch_length_m <= 0.0) project.pitch_length_m = 1.0;
    if (project.pitch_width_m <= 0.0) project.pitch_width_m = 1.0;
    if (ImGui::Button("105 x 68 (boisko zewnetrzne)")) {
      project.pitch_length_m = 105.0;
      project.pitch_width_m = 68.0;
    }
    ImGui::SameLine();
    if (ImGui::Button("40 x 20 (sala)")) {
      project.pitch_length_m = 40.0;
      project.pitch_width_m = 20.0;
    }
    ImGui::SameLine();
    if (ImGui::Button("36 x 18 (sala mala)")) {
      project.pitch_length_m = 36.0;
      project.pitch_width_m = 18.0;
    }
    ImGui::Checkbox("Tryb sali (znaczniki)", &project.hall_mode);
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Druzyny rozrozniane po zielonych znacznikach/koszulkach.\n"
          "TeamA / \"Zielone\" = gracze ze znacznikiem, TeamB = bez znacznika.");
    }
    if (project.hall_mode) {
      ImGui::TextColored({0.7f, 0.95f, 0.7f, 1},
                         "Tryb sali: tylko znaczniki (zielone / bez). Numery koszulek wylaczone.");
    }

    if (ImGui::Button("Add camera")) {
      const std::string path = open_video_dialog();
      if (!path.empty()) {
        CameraConfig cam;
        cam.video_path = path;
        cam.name = std::filesystem::path(path).stem().string();
        cam.motion = CameraMotion::Static;
        cam.world_coords = false;
        if (load_camera_sidecar(path, cam)) {
          status = "Wczytano ustawienia kamery z " +
                    camera_sidecar_path(path).filename().string();
        }
        project.cameras.push_back(cam);
        persist_config(project);
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
      if (ImGui::Combo("Rola kamery", &role_idx, role_labels, IM_ARRAYSIZE(role_labels))) {
        cam.role = static_cast<CameraRole>(role_idx);
        if (cam.role == CameraRole::TacticalPan) cam.motion = CameraMotion::Panning;
        save_camera_sidecar(cam.video_path, cam, nullptr);
        persist_config(project);
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Dla kamer w naroznikach wybierz odpowiedni naroznik boiska.\n"
            "Rola wplywa na kolejnosc i opis punktow kalibracji.");
      }

      int motion_idx = cam.motion == CameraMotion::Panning ? 1 : 0;
      if (ImGui::Combo("Ruch kamery", &motion_idx, "stala\0panning\0")) {
        cam.motion = motion_idx == 1 ? CameraMotion::Panning : CameraMotion::Static;
      }

      if (project.mode() != ProjectMode::TacticalOnly) {
        ImGui::Checkbox("Wspolrzedne swiata (metry)", &cam.world_coords);
        if (!cam.world_coords) {
          ImGui::TextColored({0.7f, 0.9f, 0.7f, 1},
                             "Tryb pikseli: heatmapy i trasy w px (bez kalibracji boiska).");
        }
      } else {
        cam.world_coords = true;
        ImGui::Text("Wspolrzedne swiata: wlaczone (tryb taktyczny)");
      }

      if (cam.world_coords && !cam.is_panning()) {
        if (ImGui::Button("Kalibruj boisko")) {
          cv::VideoCapture cap(cam.video_path);
          cv::Mat frame;
          if (cap.read(frame) && !frame.empty()) {
            const PitchModel calib_model(project.pitch_length_m, project.pitch_width_m);
            PitchCalibration calib;
            if (calib.calibrate_lines_interactive(frame, calib_model, cam.role)) {
              const auto calib_path =
                  camera_sidecar_path(cam.video_path).replace_extension(".calib");
              calib.save(calib_path);
              cam.calib_path = calib_path.string();
              save_camera_sidecar(cam.video_path, cam, &calib);
              persist_config(project);
              status = "Skalibrowano i zapisano obok wideo: " + calib_path.filename().string();
            }
          } else {
            status = "Nie mozna odczytac klatki z wideo: " + cam.name;
          }
        }
        ImGui::SameLine();
        const auto sidecar = camera_sidecar_path(cam.video_path);
        if (std::filesystem::exists(sidecar)) {
          ImGui::Text("Sidecar: %s", sidecar.filename().string().c_str());
        } else if (!cam.calib_path.empty()) {
          ImGui::Text("Plik: %s", cam.calib_path.c_str());
        }
      }
      if (ImGui::Button("Remove")) {
        project.cameras.erase(project.cameras.begin() + i);
        persist_config(project);
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
    ImGui::SliderFloat("Detect confidence", &project.detect_confidence, 0.10f, 0.80f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Lower = more detections (try 0.20-0.30 for top-down cameras).");
    }
    ImGui::SliderInt("Detect every N frames", &project.detect_frame_step, 1, 10);
    ImGui::SliderInt("Max detect width (px)", &project.detect_max_width, 640, 1920);
    ImGui::SliderInt("Detect input size", &project.detect_input_size, 320, 1280);
    ImGui::SliderInt("OCR every N detect frames", &project.ocr_frame_step, 5, 60);
    ImGui::InputDouble("Analyze max seconds", &project.analyze_max_seconds, 1.0, 5.0, "%.0f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip(
          "Limit analysis and annotated video export to the first N seconds of each video. "
          "0 = full length.");
    }
    if (project.analyze_max_seconds < 0.0) {
      project.analyze_max_seconds = 0.0;
    }
    ImGui::Checkbox("Write annotated MP4", &project.write_annotated_video);
    if (project.write_annotated_video) {
      ImGui::Checkbox("Compress annotated MP4 (H.264)", &project.annotated_video_compress);
      if (project.annotated_video_compress) {
        ImGui::Checkbox("Match source video bitrate", &project.annotated_video_match_source_bitrate);
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("Target size close to the original recording (recommended).");
        }
        ImGui::SliderInt("Annotated max width", &project.annotated_video_max_width, 0, 1920);
        if (ImGui::IsItemHovered()) {
          ImGui::SetTooltip("1280 is a good default. 0 = full resolution (much larger file).");
        }
        if (!project.annotated_video_match_source_bitrate) {
          ImGui::SliderInt("Video quality (CRF)", &project.annotated_video_crf, 28, 38);
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Higher CRF = smaller file, lower quality.");
          }
        }
      }
    }

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
      apply_output_dir();
      persist_config(project);
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
        ImGui::Text("Detection report: %s", last_result.detection_report_txt.string().c_str());
        ImGui::Text("Compression log: %s", last_result.compression_log.string().c_str());
        for (const auto& v : last_result.annotated_videos) {
          ImGui::Text("Video: %s", v.string().c_str());
        }
      }
    }

    ImGui::TextWrapped("Konfiguracja: %s", session_file.string().c_str());
    ImGui::TextWrapped("Projekt: %s", project_file.string().c_str());
    if (ImGui::Button("Save project")) {
      apply_output_dir();
      persist_config(project);
      status = "Zapisano konfiguracje";
    }
    ImGui::SameLine();
    if (ImGui::Button("Load project")) {
      std::optional<ProjectConfig> loaded;
      if (std::filesystem::exists(project_file)) {
        loaded = load_project_config(project_file);
      }
      if (!loaded) {
        loaded = load_project_config(session_file);
      }
      if (loaded) {
        project = *loaded;
        std::strncpy(output_dir, project.output_dir.c_str(), sizeof(output_dir) - 1);
        output_dir[sizeof(output_dir) - 1] = '\0';
        persist_config(project);
        status = "Wczytano konfiguracje";
      } else {
        status = "Failed to load config";
      }
    }

    const bool item_active = ImGui::IsAnyItemActive();
    if (item_was_active && !item_active) {
      apply_output_dir();
      persist_config(project);
    }
    item_was_active = item_active;

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

  apply_output_dir();
  persist_config(project);

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
