# Player Track v0.2

Multi-camera football player analytics: GUI project editor, pitch line calibration, jersey OCR + Re-ID identity fusion, PDF reports, and annotated MP4 output.

## Build (Windows)

1. Install [vcpkg](https://github.com/microsoft/vcpkg) and set `VCPKG_ROOT`.
2. Open **x64 Native Tools Command Prompt for VS 2022**.
3. Configure and build (installs OpenCV, nlohmann-json, libharu, imgui, glfw3):

```bat
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Binary: `build\Release\player_track.exe`

Download YOLO model:

```bat
powershell -ExecutionPolicy Bypass -File scripts\download_model.ps1
```

Optional jersey OCR model path: `models/jersey_ocr.onnx` (heuristic fallback used if missing).

## GUI mode (default)

Run without arguments to open the project editor:

```bat
player_track
player_track --gui
```

1. **Add camera** — pick MP4/AVI files (corners, sides, or tactical pan).
2. Set **offset (s)**, **role**, **motion** (`static` / `panning`).
3. **Calibrate** each static camera using pitch landmark clicks (min 4 points).
4. **Run analysis** — generates:
   - `results/report.pdf` — per-player stats + heatmaps
   - `results/tracks_global.csv`
   - `results/<camera>_annotated.mp4`
   - `results/heatmaps/player_N.png`

Project files save/load as `project.ptproj` (JSON).

## Project modes

| Mode | Cameras |
|------|---------|
| StaticOnly | 1–4 static cameras |
| StaticPlusTactical | Static + optional tactical (tactical helps OCR/Re-ID) |
| TacticalOnly | Single panning tactical cam (dynamic calibration, lower accuracy) |

## CLI — legacy single-camera

Original single-video tracking still works:

```bat
player_track --input run.mp4 --output track.csv --preview
player_track --mode auto --input run.mp4 --calibrate pitch.calib --pitch-rect 40.32,16.5
```

## CLI — multi-camera project

```bat
player_track --project match.ptproj
```

Example `match.ptproj`:

```json
{
  "pitch": { "length_m": 105.0, "width_m": 68.0 },
  "output_dir": "results/match_01",
  "cameras": [
    { "name": "corner_left", "path": "cam1.mp4", "role": "corner_tl", "motion": "static", "offset_s": 0.0, "world_coords": true, "calib": "results/corner_left.calib" },
    { "name": "tactical", "path": "tactical.mp4", "role": "tactical_pan", "motion": "panning", "offset_s": 1.2, "world_coords": false }
  ]
}
```

## PDF report contents

Per player:
- Total distance, average/max speed
- Sprint count and sprint distance (default threshold 7.0 m/s)
- Active time %, defensive/middle/attacking third %
- High-intensity runs, pitch heatmap (PNG embedded)
- Methodology page with identity/calibration confidence; disclaimer for TacticalOnly mode
