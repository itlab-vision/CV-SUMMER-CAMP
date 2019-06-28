#pragma once
#include <string>

struct DetectedObject
{
	int xLeft;
	int xRight;
	int yTop;
    int yBottom;
	int uuid;
	std::string classname;
};