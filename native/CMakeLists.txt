cmake_minimum_required(VERSION 3.16)
project(scamguard_gguf_verdict LANGUAGES C CXX)

if(NOT LLAMA_CPP_DIR)
  message(FATAL_ERROR "LLAMA_CPP_DIR must point to the pinned llama.cpp checkout")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp-build EXCLUDE_FROM_ALL)

add_executable(scamguard-gguf-verdict gguf_verdict_runner.cpp)
target_link_libraries(scamguard-gguf-verdict PRIVATE llama ggml)
target_compile_features(scamguard-gguf-verdict PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(scamguard-gguf-verdict PRIVATE /W4)
else()
  target_compile_options(scamguard-gguf-verdict PRIVATE -Wall -Wextra -Wpedantic)
endif()
