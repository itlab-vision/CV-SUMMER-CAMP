#include <iostream>
#include "tracker.h"

int main(int argc, char** argv)
{
	Mat source = Mat(600, 600, 1);


	Vec2i rect1start = Vec2i(0, 0);
	Vec2i rect2start = Vec2i(600, 600);
	Vec2i rect3start = Vec2i(300, 300);

	vector<DetectedObject> objects;
	DetectedObject object;
	object.xLeft = rect1start[0]- 10;
	object.xRight = rect1start[0] + 10;
	object.yTop = rect1start[1] + 10;
	object.yBottom = rect1start[1] - 10;
	object.uuid = 0;
	objects.push_back(object);
	object.xLeft = rect2start[0] - 10;
	object.xRight = rect2start[0] + 10;
	object.yTop = rect2start[1] + 10;
	object.yBottom = rect2start[1] - 10;
	object.uuid = 1;
	objects.push_back(object);
	object.xLeft = rect3start[0] - 20;
	object.xRight = rect3start[0] + 20;
	object.yTop = rect3start[1] + 10;
	object.yBottom = rect3start[1] - 10;
	object.uuid = 2;
	objects.push_back(object);

	HungarianTracker tracker;

	tracker.init(objects);
	Scalar colors[3] = { Scalar(0, 255, 255), Scalar(255, 255, 0), Scalar(0, 0, 0) };

	for (int i = 0; i < 1; i++)
	{
		Vec2i center = Vec2i(rand() % 600, rand() % 500);
		objects[0].xLeft = center[0] - 10;
		objects[0].xRight = center[0] + 10;
		objects[0].yTop = center[1] + 10;
		objects[0].yBottom = center[1] - 10;
		
		center = Vec2i(rand() % 600, rand() % 500);
		objects[1].xLeft = center[0] - 10;
		objects[1].xRight = center[0] + 50;
		objects[1].yTop = center[1] + 50;
		objects[1].yBottom = center[1] - 10;

		center = Vec2i(rand() % 600, rand() % 500);
		objects[2].xLeft = center[0] - 20;
		objects[2].xRight = center[0] + 20;
		objects[2].yTop = center[1] + 10;
		objects[2].yBottom = center[1] - 10;

		vector<DetectedObject> v = tracker.update(objects);
		

		Rect r = Rect(
			v[0].xLeft,
			v[0].yBottom,
			v[0].xRight - v[0].xLeft,
			v[0].yTop- v[0].yBottom);
		rectangle(source, r, colors[v[0].uuid % 3], 2, 8, 0);

		r = Rect(
			v[1].xLeft,
			v[1].yBottom,
			v[1].xRight - v[1].xLeft,
			v[1].yTop - v[1].yBottom);
		rectangle(source, r, colors[v[1].uuid % 3], 2, 8, 0);

		r = Rect(
			v[2].xLeft,
			v[2].yBottom,
			v[2].xRight - v[2].xLeft,
			v[2].yTop - v[2].yBottom);
		rectangle(source, r, colors[v[2].uuid % 3], 2, 8, 0);
	}
	for (int i = 0; i < tracker.trackedObjects[0].path.size() - 1; i++)
	{
		line(source, tracker.trackedObjects[0].path[i], tracker.trackedObjects[0].path[i + 1], colors[tracker.trackedObjects[0].uuid % 3], 1);
		line(source, tracker.trackedObjects[1].path[i], tracker.trackedObjects[1].path[i + 1], colors[tracker.trackedObjects[1].uuid % 3], 1);
		line(source, tracker.trackedObjects[2].path[i], tracker.trackedObjects[2].path[i + 1], colors[tracker.trackedObjects[2].uuid % 3], 1);
	}

	imshow("detectors", source);

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}