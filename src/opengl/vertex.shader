R""(
#version 330 core
layout (location = 0) in vec3 inPos;
uniform float scaleFactor;
out vec4 fragCol;
void main()
{
    vec2 scaledRes = scaleFactor * vec2(1920.0, 1080.0);
    gl_Position = vec4(inPos.x / scaledRes.x, inPos.y / scaledRes.y, inPos.z, 1.0);
    gl_PointSize = 10.0;
    fragCol = vec4(0.9f, 0.9f, 0.0f, 1.0f);
};
)""