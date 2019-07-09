#pragma once
#include <string>

struct DetectedObject
{
    int Left;
    int Right;
    int Top;
    int Bottom;
    double score;
	int classid;
    std::string classname;
};