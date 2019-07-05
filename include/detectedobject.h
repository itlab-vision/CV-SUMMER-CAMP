#pragma once
#include <string>

struct DetectedObject
{
    int xLeftBottom;
    int yLeftBottom;
    int xRightTop;
    int yRightTop;
    int uuid;
	double score;
    std::string classname;
};