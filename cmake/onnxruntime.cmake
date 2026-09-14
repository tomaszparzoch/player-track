set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root directory")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found in ${ORT_ROOT}. "
    "Run: powershell -File scripts/download_onnxruntime.ps1")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
  IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

function(copy_onnxruntime_runtime target)
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ORT_ROOT}/bin"
            "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying ONNX Runtime runtime DLLs"
  )
endfunction()
