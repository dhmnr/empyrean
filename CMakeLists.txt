# The MiniCppStarter
#
# Build and run by calling: 
# cmake -S. -Bbuild && cmake --build build && ./build/Empyrean

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ---- Project ----

project(Empyrean LANGUAGES C CXX CUDA)

enable_language(CUDA)

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

find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)

find_package(glm REQUIRED)

find_package(CUDAToolkit REQUIRED)
# ---- Create executable ----
include(cmake/SourcesFiles.cmake)
# add your source files here
add_executable(${PROJECT_NAME} ${EXE_SOURCES})
# link include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# link to your dependencies' targets here

target_link_libraries(${PROJECT_NAME} 
  CUDA::cudart
  OpenGL::GL
  glfw
  glm::glm
)
# setup your target's properties
set_target_properties(Empyrean PROPERTIES CXX_STANDARD 17)
