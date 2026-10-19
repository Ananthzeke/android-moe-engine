cmake_minimum_required(VERSION 3.22)
project(android_moe_engine VERSION 0.1.0 LANGUAGES CXX)
include(CTest)
option(AME_SANITIZERS "Enable host address and undefined behavior sanitizers" OFF)
option(AME_INFERENCE_INTEGRATION "Build pinned-llama.cpp inference gates and harness" OFF)
set(AME_LLAMA_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source tree")
set(AME_LLAMA_HOST_LIB_DIR "" CACHE PATH "Directory containing host libllama.a and ggml libs")
set(AME_LLAMA_LIBRARIES "" CACHE STRING "Explicit host llama/ggml static libraries")

function(ame_project_target target)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror)
    if(AME_SANITIZERS)
      if(ANDROID)
        message(FATAL_ERROR "AME_SANITIZERS is a host-only configuration")
      endif()
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  else()
    message(FATAL_ERROR "Project warning policy requires GCC or Clang")
  endif()
endfunction()

add_library(ame_model STATIC src/model/file_extent.cpp src/model/expert_layout.cpp)
target_include_directories(ame_model PUBLIC src/model src/common)
ame_project_target(ame_model)

add_library(ame_residency STATIC src/residency/file_source.cpp src/residency/store.cpp)
target_include_directories(ame_residency PUBLIC src/residency src/model src/common)
target_link_libraries(ame_residency PRIVATE ame_model)
ame_project_target(ame_residency)

add_library(ame_probe STATIC src/probe/probe.cpp)
target_include_directories(ame_probe PUBLIC src/probe)
ame_project_target(ame_probe)

add_library(ame_compare STATIC src/compare/compare.cpp)
target_include_directories(ame_compare PUBLIC src/compare src/common)
ame_project_target(ame_compare)

add_library(ame_prefill STATIC src/runtime/prefill_plan.cpp)
target_include_directories(ame_prefill PUBLIC src/runtime src/common)
ame_project_target(ame_prefill)

add_library(ame_run STATIC src/runtime/run_supervisor.cpp)
target_include_directories(ame_run PUBLIC src/runtime src/common)
ame_project_target(ame_run)

add_executable(ame apps/cli/main.cpp)
target_link_libraries(ame PRIVATE ame_model ame_probe ame_compare ame_run)
ame_project_target(ame)

if(AME_INFERENCE_INTEGRATION)
  if(NOT EXISTS "${AME_LLAMA_SOURCE_DIR}/include/llama.h" OR
     NOT EXISTS "${AME_LLAMA_SOURCE_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR "AME_LLAMA_SOURCE_DIR must point at the pinned llama.cpp tree")
  endif()
  if(AME_LLAMA_LIBRARIES STREQUAL "")
    if(AME_LLAMA_HOST_LIB_DIR STREQUAL "")
      message(FATAL_ERROR "Set AME_LLAMA_LIBRARIES or AME_LLAMA_HOST_LIB_DIR")
    endif()
    set(AME_LLAMA_LIBRARIES
        "${AME_LLAMA_HOST_LIB_DIR}/src/libllama.a;${AME_LLAMA_HOST_LIB_DIR}/ggml/src/libggml.a;${AME_LLAMA_HOST_LIB_DIR}/ggml/src/libggml-base.a;${AME_LLAMA_HOST_LIB_DIR}/ggml/src/libggml-cpu.a")
  endif()

  add_library(ame_backend STATIC src/backend/moe_stream.cpp)
  target_include_directories(ame_backend PUBLIC src/backend src/model src/residency src/common
                                             "${AME_LLAMA_SOURCE_DIR}/include"
                                             "${AME_LLAMA_SOURCE_DIR}/ggml/include")
  target_link_libraries(ame_backend PUBLIC ame_model ame_residency)
  ame_project_target(ame_backend)

  add_library(ame_backend_broken STATIC src/backend/moe_stream.cpp)
  target_include_directories(ame_backend_broken PUBLIC src/backend src/model src/residency src/common
                                                   "${AME_LLAMA_SOURCE_DIR}/include"
                                                   "${AME_LLAMA_SOURCE_DIR}/ggml/include")
  target_link_libraries(ame_backend_broken PUBLIC ame_model ame_residency)
  target_compile_definitions(ame_backend_broken PRIVATE AME_TEST_BREAK_MATERIALIZER=1)
  ame_project_target(ame_backend_broken)

  function(ame_inference_target target)
    target_include_directories(${target} PRIVATE "${AME_LLAMA_SOURCE_DIR}/include"
                                                "${AME_LLAMA_SOURCE_DIR}/ggml/include")
    target_link_libraries(${target} PRIVATE -Wl,--start-group ${AME_LLAMA_LIBRARIES} -Wl,--end-group pthread dl m)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
                                               CXX_EXTENSIONS NO)
    if(AME_SANITIZERS)
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endfunction()

  add_executable(moe_stream_gate bench/moe_stream_gate.cpp)
  target_link_libraries(moe_stream_gate PRIVATE ame_backend ame_prefill)
  ame_inference_target(moe_stream_gate)

  add_executable(moe_stream_gate_broken bench/moe_stream_gate.cpp)
  target_link_libraries(moe_stream_gate_broken PRIVATE ame_backend_broken)
  ame_inference_target(moe_stream_gate_broken)

  add_executable(moe_kernel_gate bench/moe_kernel_gate.cpp)
  target_link_libraries(moe_kernel_gate PRIVATE ame_residency)
  ame_inference_target(moe_kernel_gate)

  add_executable(llama_baseline_stream bench/llama_baseline.cpp)
  target_link_libraries(llama_baseline_stream PRIVATE ame_backend ame_prefill)
  ame_inference_target(llama_baseline_stream)

  add_executable(manifest_gate bench/manifest_gate.cpp)
  target_link_libraries(manifest_gate PRIVATE ame_backend)
  ame_inference_target(manifest_gate)
endif()

if(BUILD_TESTING)
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Disable BUILD_TESTING for cross-compilation")
  endif()
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  add_executable(file_extent_test tests/file_extent_test.cpp)
  target_include_directories(file_extent_test PRIVATE src/model)
  target_link_libraries(file_extent_test PRIVATE ame_model)
  ame_project_target(file_extent_test)
  add_test(NAME file_extent COMMAND file_extent_test)

  add_executable(probe_test tests/probe_test.cpp)
  target_link_libraries(probe_test PRIVATE ame_probe)
  ame_project_target(probe_test)
  add_test(NAME probe COMMAND probe_test)

  add_executable(residency_test tests/residency_test.cpp)
  target_link_libraries(residency_test PRIVATE ame_residency)
  ame_project_target(residency_test)
  add_test(NAME residency COMMAND residency_test)

  add_executable(compare_test tests/compare_test.cpp)
  target_link_libraries(compare_test PRIVATE ame_compare)
  ame_project_target(compare_test)
  add_test(NAME compare COMMAND compare_test)

  add_executable(prefill_plan_test tests/prefill_plan_test.cpp)
  target_link_libraries(prefill_plan_test PRIVATE ame_prefill)
  ame_project_target(prefill_plan_test)
  add_test(NAME prefill_plan COMMAND prefill_plan_test)

  add_executable(run_supervisor_test tests/run_supervisor_test.cpp)
  target_link_libraries(run_supervisor_test PRIVATE ame_run)
  ame_project_target(run_supervisor_test)
  add_test(NAME run_supervisor COMMAND run_supervisor_test)
  add_test(NAME repository_validation COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_scaffold_test.py)
  add_test(NAME run_record COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_record_test.py)
  add_test(NAME gguf_index COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/gguf_index_test.py)
  add_test(NAME cli COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.py $<TARGET_FILE:ame>)
endif()