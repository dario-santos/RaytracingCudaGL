#version 330 core

uniform usampler2D tex;

in vec3 ourColor;
in vec2 ourTexCoord;

out vec4 color;

void main()
{
  vec4 c = texture(tex, ourTexCoord);
  color = c / 255.0;
}
