R""(
#version 330 core
in vec4 fragCol;
out vec4 FragColor;
void main()
{
   float r = 0.0, delta = 0.0, alpha = 1.0;
   vec2 cxy = 2.0 * gl_PointCoord - 1.0;
    r = dot(cxy, cxy);
    if (r > 1.0) {
        discard;
    }
#ifdef GL_OES_standard_derivatives
    delta = fwidth(r);
    alpha = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, r);
#endif

    FragColor = alpha * fragCol;

};
)""