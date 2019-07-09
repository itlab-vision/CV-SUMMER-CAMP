#pragma once
#include <string>

struct DetectedObject
{
    float Left;
	float Top;
    float Right;    
    float Bottom;
    int uuid;
    std::string classname;
};

/*
1
2
3
left
top
right
bottom
*/