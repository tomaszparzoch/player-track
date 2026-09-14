cmake_minimum_required(VERSION 3.20)
project(player_track LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio video tracking dnn calib3d imgcodecs)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-libharu CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/onnxruntime.cmake)

set(CORE_SOURCES
  src/app_options.cpp
  src/onnx_infer.cpp
  src/person_detector.cpp
  src/selection_ui.cpp
  src/tracking_session.cpp
  src/player_tracker.cpp
  src/kinematics.cpp
  src/csv_writer.cpp
  src/pitch_calibration.cpp
  src/heatmap.cpp
  src/camera_config.cpp
  src/project_config.cpp
  src/pitch_model.cpp
  src/global_track.cpp
  src/path_utils.cpp
  src/dynamic_pitch_calibrator.cpp
  src/tracklet_tracker.cpp
  src/reid_extractor.cpp
  src/jersey_ocr.cpp
  src/identity_graph.cpp
  src/player_stats.cpp
  src/pitch_heatmap.cpp
  src/multi_camera_session.cpp
  src/pdf_report.cpp
  src/gui_app.cpp
)

add_executable(player_track
  src/main.cpp
  ${CORE_SOURCES}
)

target_include_directories(player_track PRIVATE include)
target_link_libraries(player_track PRIVATE
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  unofficial::libharu::hpdf
  glfw
  imgui::imgui
  onnxruntime::onnxruntime
)

copy_onnxruntime_runtime(player_track)

if(WIN32)
  target_link_libraries(player_track PRIVATE opengl32)
endif()

if(MSVC)
  target_compile_options(player_track PRIVATE /W4)
else()
  target_compile_options(player_track PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(make_test_video tools/make_test_video.cpp)
target_link_libraries(make_test_video PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(make_test_video PRIVATE /W4)
endif()

add_executable(make_gap_demo_video tools/make_gap_demo_video.cpp)
target_link_libraries(make_gap_demo_video PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(make_gap_demo_video PRIVATE /W4)
endif()

add_executable(grab_frames tools/grab_frames.cpp)
target_link_libraries(grab_frames PRIVATE ${OpenCV_LIBS})

add_executable(test_detector tools/test_detector.cpp src/person_detector.cpp src/onnx_infer.cpp)
target_include_directories(test_detector PRIVATE include)
target_link_libraries(test_detector PRIVATE ${OpenCV_LIBS} onnxruntime::onnxruntime)
copy_onnxruntime_runtime(test_detector)
if(MSVC)
  target_compile_options(test_detector PRIVATE /W4)
endif()
