set(SOURCES
  src/glad.c
  src/engine/common.cpp
  src/engine/body.cpp
  src/engine/nbody_engine.cpp
  src/opengl/renderer.cpp
  # src/engine/real_vector.cpp
)

set(EXE_SOURCES
  src/main.cpp
  ${SOURCES}
)
