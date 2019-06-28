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
	//virtual vector<DetectedObject> update(vector<DetectedObject> notTracked);
};

class HungarianTracker : public Tracker
{
private:
	void printMat(Mat m)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				cout << m.at<int>(i, j) << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
	void printMat2(Mat m)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				cout << m.at<Vec2i>(i, j)[0] << m.at<Vec2i>(i, j)[1] << " ";
			}
			cout << "\n";
		}
		cout << zs[0] << zs[1] << "\n";
		cout << zt[0] << zt[1] << "\n";
	}

	Vec2i calcCenter(DetectedObject rect)
	{
		return Vec2i((rect.xLeft + rect.xRight) / 2.0, (rect.yTop + rect.yBottom) / 2.0);
	}
	float calcDist(DetectedObject a, Vec2i b)
	{
		return norm(calcCenter(a), b);
	}

	void prepareDistanceMatrix(vector<DetectedObject> notTracked)
	{
		int size = notTracked.size();
		a = Mat(size, size, CV_32SC1);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				a.at<int>(i, j) = calcDist(notTracked[i], trackedObjects[j].path[trackedObjects[j].path.size()-1]);
			}

		printMat(a);
	}

	// Algorithm variables
	Mat a; // distances
	Mat c; // candidates
	Mat f; // assignment matrix
	Mat m; // depth search matrix
	vector<int> ws, wt, zs, zt; // additional vectors
	int size;

	void initPotentials()
	{
		ws = vector<int>(size);
		wt = vector<int>(size);
		zs = vector<int>(size);
		zt = vector<int>(size);

		for (int i = 0; i < size; i++)
		{
			int min = a.at<int>(i, 0);
			for (int j = 1; j < size; j++)
				if (a.at<int>(i, j) < min)
					min = a.at<int>(i, j);
			ws.at(i) = min;
		}

		f = Mat(size, size, CV_32SC1);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				f.at<int>(i, j) = 0;
	}

	void fillCandidates()
	{
		c = Mat(size, size, CV_32SC1);
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				if ( (a.at<int>(i, j) - ws[i] - wt[j]) == 0) 
					c.at<int>(i, j) = 1;
				else
					c.at<int>(i, j) = 0;
		printMat(c);
	}

	bool checkSolution()
	{
		bool ok = true;
		for (int i = 0; i < size; i++)
		{
			float strsum = 0;
			for (int j = 0; j < size; j++)
				strsum += f.at<int>(i, j);
			if (strsum != 1)
				ok = false;
		}
		for (int i = 0; i < size; i++)
		{
			float strsum = 0;
			for (int j = 0; j < size; j++)
				strsum += f.at<int>(i, j);
			if (strsum != 1)
				ok = false;
		}
		return ok;
	}

	bool greedyInit()
	{
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				// если в столбце j матрицы F пока не было значений У1Ф, то F(i, j) : = 1
				if (c.at<int>(i, j) == 1)
				{
					float sum = 0;
					for (int k = 0; k < size; k++)
						sum += f.at<int>(k, j);
					if (sum == 0)
						f.at<int>(i, j) = 1;
				}
			}
		cout << "Greedy";
		printMat(f);

		return checkSolution();
	}
	
	Vec2i fillMatrixM(int prev_i, int prev_j)
	{
		cout << "M";
		printMat2(m);
		int i = prev_i;
		for (int j = 0; j < size; j++)
			if (c.at<int>(i, j) == 1 && zt[j] == 0)
			{
				Vec2i & p = m.at<Vec2i>(i, j);
				p[0] = prev_i;
				p[1] = prev_j;
				zt[j] = 1;

				int newi = -1;
				for (int k = 0; k < 3; k++)
					if (f.at<int>(k, j) == 1 && k != i)
					{ newi = k; break; }
				if (newi == -1)
					return Vec2i(i, j);

				zs[newi] = 1;
				Vec2i & p2 = m.at<Vec2i>(newi, j);
				p2[0] = i;
				p2[1] = j;
				
				Vec2i tmp = fillMatrixM(newi, j);

				

				if (tmp[0] >= 0 && tmp[1] >= 0)
					return tmp;
			}

		return Vec2i(-1, -1);
	}

	void doOverload(int res_i, int res_j)
	{
		int i = res_i;
		int j = res_j;

		while (i >= 0 && j >= 0)
		{
			f.at<int>(i,j) = 1 - f.at<int>(i, j);
			Vec2i s = m.at<Vec2i>(i, j);
			i = s[0];
			j = s[1];
		}
	}

	void mark(vector<DetectedObject> notTracked)
	{
		// use f matrix to track 
		for (int i = 0; i < size; i++)
			for(int j = 0; j < size; j++)
				if (f.at<int>(i, j) == 1)
				{
					notTracked[j].uuid = trackedObjects[i].uuid;
					trackedObjects[i].path.push_back(calcCenter(notTracked[j]));
				}
	}

	bool tryRefineWithoutChangingPotentials()
	{
		m = Mat(size, size, CV_32SC2, Vec2i(-1, -1));

		for (int ii = 0; ii < size; ii++) 
			{ zs[ii] = 0; zt[ii] = 0; }

		for (int i = 0; i < size; i++)
		{
			bool has1 = false;
			for (int j = 0; j < size; j++)
			{
				if (f.at<int>(i,j) == 1)
					has1 = true;
			}
			if (has1 == true)
				continue;

			
			Vec2i newPoint = fillMatrixM(i, -1);
			//printMat(f);
			if (newPoint[0] >= 0 && newPoint[1] >= 0)
			{
				doOverload(newPoint[0], newPoint[1]);
				m = Mat(size, size, CV_32SC2, Vec2i(-1, -1));
				for (int ii = 0; ii < size; ii++)
				{zs[ii] = 0; zt[ii] = 0;}
				return true;
			}
		}
		return false;
	}

	void changePotentials()
	{
		int d = 1000000000;
		for (int i = 0; i < size; i++)
			if (zs[i] == 1)
				for (int j = 0; j < size; j++)
					if (zt[j] == 0)
						d = min(d, a.at<int>(i,j) - ws[i] - wt[j]);
		for (int i = 0; i < size; i++)
			if (zs[i] == 1)
				ws[i] = ws[i] + d;
		for (int j = 0; j < size; j++)
			if (zt[j] == 1)
				wt[j] = wt[j] - d;
	}

	Mat stub()
	{
		Mat res = Mat(3, 3, CV_32SC1);
		res.at<int>(0, 0) = 10;
		res.at<int>(0, 1) = 20;
		res.at<int>(0, 2) = 30;
		res.at<int>(1, 0) = 30;
		res.at<int>(1, 1) = 30;
		res.at<int>(1, 2) = 30;
		res.at<int>(2, 0) = 30;
		res.at<int>(2, 1) = 30;
		res.at<int>(2, 2) = 20;

		return res;
	}

public:

	void init(vector<DetectedObject> detected)
	{
		size = detected.size();
		for (int i = 0; i < size; i++)
		{
			TrackedObject newObject;
			newObject.uuid = i;
			newObject.path.push_back(calcCenter(detected[i]));
			trackedObjects.push_back(newObject);
		}
	}

	vector<DetectedObject> update(vector<DetectedObject> notTracked)
	{
		/*size = 3;
		a = stub();*/
		prepareDistanceMatrix(notTracked);
		initPotentials();
		fillCandidates();
		bool greed = greedyInit();
		if (greed)
		{
			mark(notTracked);
			return notTracked;
		}

		while (true)
		{
			fillCandidates();
			bool is_ok = true;
			while (is_ok == true)
			{
				is_ok = tryRefineWithoutChangingPotentials();

//				printMat(c);
//				printMat(f);
			}
			if (checkSolution())
			{
				printMat(f);
				mark(notTracked);
				return notTracked;
			}

			changePotentials();
		}

		printMat(f);

		return notTracked;
	}
};

////// https://docs.opencv.org/3.4/d5/d07/tutorial_multitracker.html