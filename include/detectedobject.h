#pragma once
#include <string>

struct DetectedObject
{
    int Left;
    int Right;
    int Top;
    int Bottom;
    int uuid;
	int classid;
    std::string classname;
	float confidence;
};