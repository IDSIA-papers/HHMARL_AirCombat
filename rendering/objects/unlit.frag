#version 330 core

in vec3 Normal;

uniform vec4 color;

out vec4 FragColor;

void main() {
    FragColor = vec4(color.rgb, color.a);
}