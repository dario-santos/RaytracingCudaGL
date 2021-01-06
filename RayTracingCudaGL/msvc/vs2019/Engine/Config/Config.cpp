#include "./Config.hpp"

map<string, map<string, int>> Config::keys = {};


int Config::highscore = 0;

void Config::LoadConfig(char const* path)
{

  ifstream file;
  file.open("./config.cfg");

  if (!file.is_open())
    return;

  string line = "";
  string active = "";

  while (getline(file, line))
  {
    // Ignore Comments
    if (line == "" || line[0] == '#')
      continue;

    istringstream is_line(line);
    string key;

    if (line[0] == '[')
    {
      string section;

      getline(is_line, section, ']');

      section.replace(0, 1, "");
      active = section;

      Config::keys.insert({ section, {} });
    }
    else if (getline(is_line, key, '='))
    {
      string value;
      if (getline(is_line, value))
        Config::keys[active].insert({ key, stoi(value) });
    }
  }
  file.close();
}

void Config::LoadScore(char const* path)
{
  ifstream rf(path, ios::out | ios::binary);

  rf.read((char*)&Config::highscore, sizeof(int));

  rf.close();
}

void Config::SaveScore(char const* path)
{
  ofstream wf(path, ios::out | ios::binary);

  wf.write((char*)&Config::highscore, sizeof(Config::highscore));
  wf.close();
}
