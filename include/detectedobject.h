#pragma once
#include <string>

struct DetectedObject
{
	int classid;
	std::string className;
	float score;
	float left;
	float bottom;
	float right;
	float top;
	

	DetectedObject(int c, std::string cn, float s, float l, float b, float r, float t) : classid(c), className(cn), score(s), left(l), bottom(b), right(r), top(t)
	{
	}
};