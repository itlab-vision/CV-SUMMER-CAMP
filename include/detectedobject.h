#pragma once
#include <string>

struct DetectedObject
{
    int classId;
    float score;
    float left;
    float bottom;
    float right;
    float top;
    std::string className;

    // DetectedObject(int clId, float sc, float l,
    //   float btm, float r, float t, std::string clsName): classId(clId), score(sc),
    //   left(l), right(r), bottom(btm), top(t) {}
};
