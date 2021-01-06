#include "Input.hpp"

map<int, int> Input::keys = {
  {GLFW_KEY_0,     GLFW_RELEASE},
  {GLFW_KEY_1,     GLFW_RELEASE},
  {GLFW_KEY_2,     GLFW_RELEASE},
  {GLFW_KEY_3,     GLFW_RELEASE},
  {GLFW_KEY_4,     GLFW_RELEASE},
  {GLFW_KEY_5,     GLFW_RELEASE},
  {GLFW_KEY_6,     GLFW_RELEASE},
  {GLFW_KEY_7,     GLFW_RELEASE},
  {GLFW_KEY_8,     GLFW_RELEASE},
  {GLFW_KEY_9,     GLFW_RELEASE},
  {GLFW_KEY_A,     GLFW_RELEASE},
  {GLFW_KEY_B,     GLFW_RELEASE},
  {GLFW_KEY_C,     GLFW_RELEASE},
  {GLFW_KEY_D,     GLFW_RELEASE},
  {GLFW_KEY_E,     GLFW_RELEASE},
  {GLFW_KEY_F,     GLFW_RELEASE},
  {GLFW_KEY_G,     GLFW_RELEASE},
  {GLFW_KEY_H,     GLFW_RELEASE},
  {GLFW_KEY_I,     GLFW_RELEASE},
  {GLFW_KEY_J,     GLFW_RELEASE},
  {GLFW_KEY_K,     GLFW_RELEASE},
  {GLFW_KEY_L,     GLFW_RELEASE},
  {GLFW_KEY_M,     GLFW_RELEASE},
  {GLFW_KEY_N,     GLFW_RELEASE},
  {GLFW_KEY_O,     GLFW_RELEASE},
  {GLFW_KEY_P,     GLFW_RELEASE},
  {GLFW_KEY_Q,     GLFW_RELEASE},
  {GLFW_KEY_R,     GLFW_RELEASE},
  {GLFW_KEY_S,     GLFW_RELEASE},
  {GLFW_KEY_T,     GLFW_RELEASE},
  {GLFW_KEY_U,     GLFW_RELEASE},
  {GLFW_KEY_V,     GLFW_RELEASE},
  {GLFW_KEY_W,     GLFW_RELEASE},
  {GLFW_KEY_X,     GLFW_RELEASE},
  {GLFW_KEY_Y,     GLFW_RELEASE},
  {GLFW_KEY_Z,     GLFW_RELEASE},
  {GLFW_KEY_UP,    GLFW_RELEASE},
  {GLFW_KEY_RIGHT, GLFW_RELEASE},
  {GLFW_KEY_LEFT,  GLFW_RELEASE},
  {GLFW_KEY_DOWN,  GLFW_RELEASE},
  {GLFW_KEY_SPACE, GLFW_RELEASE},
};

float Input::GetAxis(AxesCode axis, Gamepad device)
{
  int count;
  const float* axes = glfwGetJoystickAxes(static_cast<int>(device), &count);

  return axes != NULL ? axes[static_cast<int>(axis)] : 0.0f;
}

float Input::GetAxis(string axis, Gamepad device)
{
  int count;
  const float* axes = glfwGetJoystickAxes(static_cast<int>(device), &count);

  return axes != NULL ? axes[Config::keys["Axis"][axis]] : 0.0f;
}

bool Input::GetButton(ButtonCode button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[static_cast<int>(button)] != GLFW_RELEASE : false;
}

bool Input::GetButton(string button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[Config::keys["Gamepad"][button]] != GLFW_RELEASE : false;
}

bool Input::GetButtonDown(ButtonCode button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[static_cast<int>(button)] == GLFW_PRESS : false;
}

bool Input::GetButtonDown(string button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[Config::keys["Gamepad"][button]] == GLFW_PRESS : false;
}

bool Input::GetButtonUp(ButtonCode button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[static_cast<int>(button)] == GLFW_RELEASE : false;
}

bool Input::GetButtonUp(string button, Gamepad device)
{
  int count;
  unsigned const char* buttons = glfwGetJoystickButtons(static_cast<int>(device), &count);

  return buttons != NULL ? buttons[Config::keys["Gamepad"][button]] == GLFW_RELEASE : false;
}

bool Input::GetKey(KeyCode key)
{
  return Input::keys[static_cast<int>(key)] != GLFW_RELEASE;
}

bool Input::GetKey(string key)
{
  return Input::keys[Config::keys["Keyboard"][key]] != GLFW_RELEASE;
}

bool Input::GetKeyDown(KeyCode key)
{
  return Input::keys[static_cast<int>(key)] == Status::Press;
}

bool Input::GetKeyDown(string key)
{
  return Input::keys[Config::keys["Keyboard"][key]] == GLFW_PRESS;
}

bool Input::GetKeyUp(KeyCode key)
{
  return Input::keys[static_cast<int>(key)] == GLFW_RELEASE;
}

bool Input::GetKeyUp(string key)
{
  return Input::keys[Config::keys["Keyboard"][key]] == GLFW_RELEASE;
}

void Input::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  Input::keys[key] = action;
}