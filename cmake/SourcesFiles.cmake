set(SOURCES
  src/glad.c
  src/handle_threads.cpp
  src/engine/common.cpp
  src/engine/body.cpp
  src/engine/nbody_engine.cpp
  src/opengl/renderer.cpp
  src/utils/fps_counter.cpp
  # src/engine/real_vector.cpp
)

set(EXE_SOURCES
  src/main.cpp
  ${SOURCES}
)
