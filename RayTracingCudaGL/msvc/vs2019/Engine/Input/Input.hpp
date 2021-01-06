/*********************************************************************
  * \file   Input.hpp
  * \brief  Input class, manages the input system
  *
  * \author Dário Santos
  * \date   April 2020
 ***********************************************************************/
#pragma once

#include <GLFW/glfw3.h>

#include <map>
#include <vector>
#include <string>

using namespace std;

#include <Engine/Config/Config.hpp>

enum class Gamepad {
  Gamepad1 = GLFW_JOYSTICK_1,
  Gamepad2 = GLFW_JOYSTICK_2,
  Gamepad3 = GLFW_JOYSTICK_3,
  Gamepad4 = GLFW_JOYSTICK_4,
  Gamepad5 = GLFW_JOYSTICK_5,
  Gamepad6 = GLFW_JOYSTICK_6,
  Gamepad7 = GLFW_JOYSTICK_7,
  Gamepad8 = GLFW_JOYSTICK_8,
  Gamepad9 = GLFW_JOYSTICK_9,
  Gamepad10 = GLFW_JOYSTICK_10,
  Gamepad11 = GLFW_JOYSTICK_11,
  Gamepad12 = GLFW_JOYSTICK_12,
  Gamepad13 = GLFW_JOYSTICK_13,
  Gamepad14 = GLFW_JOYSTICK_14,
  Gamepad15 = GLFW_JOYSTICK_15,
  Gamepad16 = GLFW_JOYSTICK_16,
};

enum class ButtonCode {
  A = 0,
  B,
  X,
  Y,
  LB,
  RB,
  Select,
  Start,
  L3,
  R3,
  DPAD_UP,
  DPAD_RIGHT,
  DPAD_DOWN,
  DPAD_LEFT,
};

enum class AxesCode {
  LAxis_X = 0,
  LAxis_Y,
  RAxis_X,
  RAxis_Y,
  LT,
  RT
};

enum class KeyCode {
  Alpha0 = 48,
  Alpha1 = 49,
  Alpha2 = 50,
  Alpha3 = 51,
  Alpha4 = 52,
  Alpha5 = 53,
  Alpha6 = 54,
  Alpha7 = 55,
  Alpha8 = 56,
  Alpha9 = 57,
  A = 65,
  B = 66,
  C = 67,
  D = 68,
  E = 69,
  F = 70,
  G = 71,
  H = 72,
  I = 73,
  J = 74,
  K = 75,
  L = 76,
  M = 77,
  N = 78,
  O = 79,
  P = 80,
  Q = 81,
  R = 82,
  S = 83,
  T = 84,
  U = 85,
  V = 86,
  W = 87,
  X = 88,
  Y = 89,
  Z = 90,
  RightArrow = 262,
  LeftArrow = 263,
  DownArrow = 264,
  UpArrow = 265
};

/**
 * Input
 */
class Input
{
private:
  // A hash map contains the status of the supported keys .
  static map<int, int> keys;
  enum Status { Release = 0, Press, Hold };
  Input();
  ~Input();

public:
  /**
   * GetAxis
   *
   * \param axis The axis code of the enum AxisCode
   * \param device The gamepad that will be tested
   * \return The value of the virtual axis identified by axisName. The value is in the range [-1.0, 1.0]
   */
  static float GetAxis(AxesCode axis, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetAxis
   *
   * \param axis The axis name, the available names depend of the ones in config.cfg file
   * \param device The gamepad that will be tested
   * \return The value of the virtual axis identified by axisName. The value is in the range [-1.0, 1.0]
   */
  static float GetAxis(string axis, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButton
   *
   * \param button The button code of the enum ButtonCode
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButton(ButtonCode button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButton
   *
   * \param button The button name, the available names depend of the ones in config.cfg file
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButton(string button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButtonDown
   *
   * \param button The button code of the enum ButtonCode
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButtonDown(ButtonCode button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButtonDown
   *
   * \param button The button name, the available names depend of the ones in config.cfg file
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButtonDown(string button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButtonUp
   *
   * \param button The button code of the enum ButtonCode
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButtonUp(ButtonCode button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetButtonUp
   *
   * \param button The button name, the available names depend of the ones in config.cfg file
   * \param device The gamepad that will be tested
   * \return True while the virtual button identified by button is held down
   */
  static bool GetButtonUp(string button, Gamepad device = Gamepad::Gamepad1);

  /**
   * GetKey
   *
   * \param key The key code of the enum KeyCode
   * \return True while the user holds down the key identified by name
   */
  static bool GetKey(KeyCode key);

  /**
   * GetKey
   *
   * \param key The key name, the available names depend of the ones in config.cfg file
   * \return True while the user holds down the key identified by name
   */
  static bool GetKey(string key);

  /**
   * GetKeyDown
   *
   * \param key The key code of the enum KeyCode
   * \return True in the frame the starts pressing down the key identified by name
   */
  static bool GetKeyDown(KeyCode key);

  /**
   * GetKeyDown
   *
   * \param key The key name, the available names depend of the ones in config.cfg file
   * \return True in the frame the starts pressing down the key identified by name
   */
  static bool GetKeyDown(string key);

  /**
   * GetKeyUp
   *
   * \param key The key code of the enum KeyCode
   * \return True during the frame the user releases the key identified by name
   */
  static bool GetKeyUp(KeyCode key);

  /**
   * GetKeyUp
   *
   * \param key The key name, the available names depend of the ones in config.cfg file
   * \return True during the frame the user releases the key identified by name
   */
  static bool GetKeyUp(string key);

  /**
   * KeyCallback
   *
   * \param window The GLFW active window
   * \param key The key that evoked the event
   * \param scancode The code of the pressed key
   * \param action The status of the key
   * \param mods If there are active mods, capslock, etc.
   */
  static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
};
