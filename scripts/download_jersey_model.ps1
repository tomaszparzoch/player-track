# Placeholder for optional jersey OCR ONNX model.
# Without this file the app uses heuristic jersey detection (lower accuracy).
# To add a model, place jersey_ocr.onnx in models/ and update jersey_ocr_model_path in .ptproj

Write-Host "No public jersey OCR model bundled yet."
Write-Host "The app will use heuristic fallback for jersey numbers."
Write-Host "Place a custom ONNX classifier at models/jersey_ocr.onnx if available."
