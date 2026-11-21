#version 430 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D renderTex;
uniform vec3 backgroundColor;

void main()
{
    vec4 volume = texture(renderTex, uv);

    // The ray marcher accumulates PREMULTIPLIED colour (it already scaled each
    // sample by its own alpha), so the correct way to put it on screen is
    //     result = premultiplied_rgb + (1 - alpha) * background
    // Sampling .rgb straight to the screen left every partially transparent ray
    // sitting on pure black, which made the volume look like it was floating in a void.
    vec3 composited = volume.rgb + (1.0 - volume.a) * backgroundColor;

    fragColor = vec4(composited, 1.0);
}
