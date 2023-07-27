# The MiniCppStarter
#
# Build and run by calling: 
# cmake -S. -Bbuild && cmake --build build && ./build/Empyrean

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ---- Project ----

project(Empyrean LANGUAGES CXX CUDA)

# ---- Fetch CPM ----

set(CPM_DOWNLOAD_VERSION 0.32.0) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
  file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# ---- Add dependencies via CPM ----
# See https://github.com/TheLartians/CPM.cmake for details and examples
CPMAddPackage(
  GITHUB_REPOSITORY libsdl-org/SDL
  GIT_TAG release-2.28.1
)
# CUDA Stuff
enable_language(CUDA)

find_package(CUDAToolkit)
# ---- Create executable ----

# add your source files here
add_executable(Empyrean src/main.cpp src/visualizer/sdl2_visualizer.cpp)
# link include directories
target_include_directories(Empyrean PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# link to your dependencies' targets here
target_link_libraries(${PROJECT_NAME} SDL2::SDL2)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
# setup your target's properties
set_target_properties(Empyrean PROPERTIES CXX_STANDARD 17)
