#pragma once
#include <string>

struct DetectedObject
{	
	int classid;
	std::string className;
	float confidence;
    float Left;
    float Right;
    float Top;
	float Bottom;
    
};