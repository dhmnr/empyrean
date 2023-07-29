set(SOURCES
    src/visualizer/sdl2_visualizer.cpp
    src/engine/common.cpp
    src/engine/cosmic_body.cpp
    src/engine/nbody_engine.cpp
    src/engine/real_vector.cpp
    src/engine/serial_euler.cpp
)

set(EXE_SOURCES
    src/main.cpp
    ${SOURCES}
)