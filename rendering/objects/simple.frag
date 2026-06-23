#version 330 core

in vec3 Normal;

uniform vec4 color;
uniform vec3 lightDir;

out vec4 FragColor;

void main() {
    float ambient = .2;
    FragColor = vec4(color.rgb * (ambient + max(dot(normalize(lightDir), -Normal), 0.0)), color.a);
}