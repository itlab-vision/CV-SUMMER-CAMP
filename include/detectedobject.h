#pragma once
#include <string>

struct DetectedObject
{
	float classid;
	float score;
	float left;
	float bottom;
	float right;
	float top;
	std::string name;

	DetectedObject(float c, float s, float l, float b, float r, float t,std::string cn) : classid(c), score(s), left(l),bottom(b),right(r),top(t), name(cn)
	{
	}
};

//[image_number, classid, score, left, bottom, right, top]