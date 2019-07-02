#pragma once
#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace std;
using namespace cv;

struct TrackedObject
{
	vector<Vec2i> path;
	int uuid;
};

class Tracker
{
public:
	vector <TrackedObject> trackedObjects;
	virtual vector<DetectedObject> update(vector<DetectedObject> notTracked) = 0 {}
};
