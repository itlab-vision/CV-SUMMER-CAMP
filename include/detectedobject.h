#pragma once
#include <string>

struct DetectedObject
{
    int xLeftBottom;
    int yLeftBottom;
    int xRightTop;
    int yRightTop;
    int uuid;
    std::string classname;
};