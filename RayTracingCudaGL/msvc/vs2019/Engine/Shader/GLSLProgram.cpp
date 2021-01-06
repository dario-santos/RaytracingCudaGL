#include "GLSLProgram.hpp"

#include <fstream>
using std::ifstream;
using std::ios;

#include <sstream>
#include <sys/stat.h>

namespace GLSLShaderInfo {
  struct shader_file_extension {
    const char* ext;
    GLSLShader::GLSLShaderType type;
  };

  struct shader_file_extension extensions[] =
  {
    {".vs", GLSLShader::VERTEX},
    {".vert", GLSLShader::VERTEX},
    {".gs", GLSLShader::GEOMETRY},
    {".geom", GLSLShader::GEOMETRY},
    {".tcs", GLSLShader::TESS_CONTROL},
    {".tes", GLSLShader::TESS_EVALUATION},
    {".fs", GLSLShader::FRAGMENT},
    {".frag", GLSLShader::FRAGMENT},
    {".cs", GLSLShader::COMPUTE}
  };
}

GLSLProgram::~GLSLProgram() {
  if (handle == 0) return;

  // Query the number of attached shaders
  GLint numShaders = 0;
  glGetProgramiv(handle, GL_ATTACHED_SHADERS, &numShaders);

  // Get the shader names
  GLuint* shaderNames = new GLuint[numShaders];
  glGetAttachedShaders(handle, numShaders, NULL, shaderNames);

  // Delete the shaders
  for (int i = 0; i < numShaders; i++)
    glDeleteShader(shaderNames[i]);

  // Delete the program
  glDeleteProgram(handle);

  delete[] shaderNames;
}

void GLSLProgram::compileShader(const char* fileName)
throw(GLSLProgramException) {
  int numExts = sizeof(GLSLShaderInfo::extensions) / sizeof(GLSLShaderInfo::shader_file_extension);

  // Check the file name's extension to determine the shader type
  string ext = getExtension(fileName);
  GLSLShader::GLSLShaderType type = GLSLShader::VERTEX;
  bool matchFound = false;
  for (int i = 0; i < numExts; i++) {
    if (ext == GLSLShaderInfo::extensions[i].ext) {
      matchFound = true;
      type = GLSLShaderInfo::extensions[i].type;
      break;
    }
  }

  // If we didn't find a match, throw an exception
  if (!matchFound) {
    string msg = "Unrecognized extension: " + ext;
    throw GLSLProgramException(msg);
  }

  // Pass the discovered shader type along
  compileShader(fileName, type);
}

string GLSLProgram::getExtension(const char* name) {
  string nameStr(name);

  size_t loc = nameStr.find_last_of('.');
  if (loc != string::npos) {
    return nameStr.substr(loc, string::npos);
  }
  return "";
}

void GLSLProgram::compileShader(const char* fileName,
  GLSLShader::GLSLShaderType type)
  throw(GLSLProgramException)
{
  if (!fileExists(fileName))
  {
    string message = string("Shader: ") + fileName + " not found.";
    throw GLSLProgramException(message);
  }

  if (handle <= 0) {
    handle = glCreateProgram();
    if (handle == 0) {
      throw GLSLProgramException("Unable to create shader program.");
    }
  }

  ifstream inFile(fileName, ios::in);
  if (!inFile) {
    string message = string("Unable to open: ") + fileName;
    throw GLSLProgramException(message);
  }

  // Get file contents
  std::stringstream code;
  code << inFile.rdbuf();
  inFile.close();

  compileShader(code.str(), type, fileName);
}

void GLSLProgram::compileShader(const string& source,
  GLSLShader::GLSLShaderType type,
  const char* fileName)
  throw(GLSLProgramException)
{
  if (handle <= 0) {
    handle = glCreateProgram();
    if (handle == 0) {
      throw GLSLProgramException("Unable to create shader program.");
    }
  }

  GLuint shaderHandle = glCreateShader(type);

  const char* c_code = source.c_str();
  glShaderSource(shaderHandle, 1, &c_code, NULL);

  // Compile the shader
  glCompileShader(shaderHandle);

  // Check for errors
  int result;
  glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &result);
  if (GL_FALSE == result) {
    // Compile failed, get log
    int length = 0;
    string logString;
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
      char* c_log = new char[length];
      int written = 0;
      glGetShaderInfoLog(shaderHandle, length, &written, c_log);
      logString = c_log;
      delete[] c_log;
    }
    string msg;
    if (fileName) {
      msg = string(fileName) + ": shader compliation failed\n";
    }
    else {
      msg = "Shader compilation failed.\n";
    }
    msg += logString;

    throw GLSLProgramException(msg);

  }
  else {
    // Compile succeeded, attach shader
    glAttachShader(handle, shaderHandle);
  }
}

void GLSLProgram::link() throw(GLSLProgramException)
{
  if (linked) return;
  if (handle <= 0)
    throw GLSLProgramException("Program has not been compiled.");

  glLinkProgram(handle);

  int status = 0;
  glGetProgramiv(handle, GL_LINK_STATUS, &status);
  if (GL_FALSE == status) {
    // Store log and return false
    int length = 0;
    string logString;

    glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &length);

    if (length > 0) {
      char* c_log = new char[length];
      int written = 0;
      glGetProgramInfoLog(handle, length, &written, c_log);
      logString = c_log;
      delete[] c_log;
    }

    throw GLSLProgramException(string("Program link failed:\n") + logString);
  }
  else {
    uniformLocations.clear();
    linked = true;
  }
}

void GLSLProgram::use()
{
  glUseProgram(handle);
}

int GLSLProgram::getHandle()
{
  return handle;
}

bool GLSLProgram::isLinked()
{
  return linked;
}

void GLSLProgram::bindAttribLocation(GLuint location, const char* name)
{
  glBindAttribLocation(handle, location, name);
}

void GLSLProgram::bindFragDataLocation(GLuint location, const char* name)
{
  glBindFragDataLocation(handle, location, name);
}

void GLSLProgram::setUniform(const char* name, float x, float y, float z)
{
  GLint loc = getUniformLocation(name);
  glUniform3f(loc, x, y, z);
}

void GLSLProgram::setUniform(const char* name, const vec3& v)
{
  this->setUniform(name, v.x, v.y, v.z);
}

void GLSLProgram::setUniform(const char* name, const vec4& v)
{
  GLint loc = getUniformLocation(name);
  glUniform4f(loc, v.x, v.y, v.z, v.w);
}

void GLSLProgram::setUniform(const char* name, const vec2& v)
{
  GLint loc = getUniformLocation(name);
  glUniform2f(loc, v.x, v.y);
}

void GLSLProgram::setUniform(const char* name, const mat4& m)
{
  GLint loc = getUniformLocation(name);
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void GLSLProgram::setUniform(const char* name, const mat3& m)
{
  GLint loc = getUniformLocation(name);
  glUniformMatrix3fv(loc, 1, GL_FALSE, &m[0][0]);
}

void GLSLProgram::setUniform(const char* name, float val)
{
  GLint loc = getUniformLocation(name);
  glUniform1f(loc, val);
}

void GLSLProgram::setUniform(const char* name, int val)
{
  GLint loc = getUniformLocation(name);
  glUniform1i(loc, val);
}

void GLSLProgram::setUniform(const char* name, GLuint val)
{
  GLint loc = getUniformLocation(name);
  glUniform1ui(loc, val);
}

void GLSLProgram::setUniform(const char* name, bool val)
{
  int loc = getUniformLocation(name);
  glUniform1i(loc, val);
}

const char* GLSLProgram::getTypeString(GLenum type) {
  // There are many more types than are covered here, but
  // these are the most common in these examples.
  switch (type) {
  case GL_FLOAT:
    return "float";
  case GL_FLOAT_VEC2:
    return "vec2";
  case GL_FLOAT_VEC3:
    return "vec3";
  case GL_FLOAT_VEC4:
    return "vec4";
  case GL_DOUBLE:
    return "double";
  case GL_INT:
    return "int";
  case GL_UNSIGNED_INT:
    return "unsigned int";
  case GL_BOOL:
    return "bool";
  case GL_FLOAT_MAT2:
    return "mat2";
  case GL_FLOAT_MAT3:
    return "mat3";
  case GL_FLOAT_MAT4:
    return "mat4";
  default:
    return "?";
  }
}

void GLSLProgram::validate() throw(GLSLProgramException)
{
  if (!isLinked())
    throw GLSLProgramException("Program is not linked");

  GLint status;
  glValidateProgram(handle);
  glGetProgramiv(handle, GL_VALIDATE_STATUS, &status);

  if (GL_FALSE == status) {
    // Store log and return false
    int length = 0;
    string logString;

    glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &length);

    if (length > 0) {
      char* c_log = new char[length];
      int written = 0;
      glGetProgramInfoLog(handle, length, &written, c_log);
      logString = c_log;
      delete[] c_log;
    }

    throw GLSLProgramException(string("Program failed to validate\n") + logString);

  }
}

int GLSLProgram::getUniformLocation(const char* name)
{
  std::map<string, int>::iterator pos;
  pos = uniformLocations.find(name);

  if (pos == uniformLocations.end()) {
    uniformLocations[name] = glGetUniformLocation(handle, name);
  }

  return uniformLocations[name];
}

bool GLSLProgram::fileExists(const string& fileName)
{
  struct stat info;
  int ret = -1;

  ret = stat(fileName.c_str(), &info);
  return 0 == ret;
}
