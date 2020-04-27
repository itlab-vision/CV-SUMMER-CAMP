#pragma once
#include <string>

struct DetectedObject
{
    int Left;
    int Right;
    int Top;
    int Bottom;
    int classId;
	double score;
    std::string classname;
};