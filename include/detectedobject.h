#pragma once
#include <string>

struct DetectedObject
{
    int Left;
	int Top;
    int Right;    
    int Bottom;
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