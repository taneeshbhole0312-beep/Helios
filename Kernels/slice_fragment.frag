#version 430 core

in vec2 uv;

out vec4 FragColor;

uniform sampler2D curlSliceTexture;

uniform float curl_min;
uniform float curl_max;


vec3 curlColor(float t)
{
    t = clamp(t, 0.0, 1.0);

    // Color stops
    vec3 c0 = vec3(0.00, 0.00, 0.12); // dark blue
    vec3 c1 = vec3(0.00, 0.15, 1.00); // blue
    vec3 c2 = vec3(0.00, 0.85, 1.00); // cyan
    vec3 c3 = vec3(0.00, 1.00, 0.20); // green
    vec3 c4 = vec3(1.00, 1.00, 0.00); // yellow
    vec3 c5 = vec3(1.00, 0.45, 0.00); // orange
    vec3 c6 = vec3(1.00, 0.00, 0.00); // red
    vec3 c7 = vec3(1.00, 1.00, 1.00); // white

    float p = t * 7.0;

    int segment = int(floor(p));
    float f = fract(p);

    // Smoothstep interpolation
    f = f * f * (3.0 - 2.0 * f);

    if (segment == 0)
        return mix(c0, c1, f);

    if (segment == 1)
        return mix(c1, c2, f);

    if (segment == 2)
        return mix(c2, c3, f);

    if (segment == 3)
        return mix(c3, c4, f);

    if (segment == 4)
        return mix(c4, c5, f);

    if (segment == 5)
        return mix(c5, c6, f);

    return mix(c6, c7, f);
}


void main()
{
    float curl = texture(curlSliceTexture, uv).r;

    float range = curl_max - curl_min;
    float t = (range != 0.0) ? (curl - curl_min) / range : 0.5;

    vec3 color = curlColor(t);

    FragColor = vec4(color, 1.0);
}