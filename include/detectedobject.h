#pragma once
#include <string>

struct DetectedObject
{
	int Score;
    int Left;
    int Right;
    int Top;
    int Bottom;
    int uuid;
    std::string classname;
};