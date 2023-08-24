R""(
#version 330 core
layout (location = 0) in vec3 inPos;
uniform float scaleFactor;
void main()
{
    vec2 scaledRes = scaleFactor * vec2(1920.0, 1080.0);
    gl_Position = vec4(inPos.x / scaledRes.x, inPos.y / scaledRes.y, inPos.z, 1.0);
    gl_PointSize = 3.0;
};
)""