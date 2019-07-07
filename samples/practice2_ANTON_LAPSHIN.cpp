#include <iostream>
#include <fstream>
#include <string>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             |         | image to process                        }"
"{ w  width                             |         | image width for classification          }"
"{ h  heigth                            |         | image heigth for classification         }"
"{ model_path                           |         | path to model                           }"
"{ config_path                          |         | path to model configuration             }"
"{ label_path                           |         | path to class labels                    }"
"{ mean                                 | 0 0 0 0 | vector of mean model values             }"
"{ swap                                 | <FALSE> | swap R and B channels. TRUE|FALSE       }"
"{ boundaries                           |         | array of area coordinates (x1 y1 x2 y2) }"
"{ top                                  |         | output top n results                    }"
"{ q ? help usage                       |         | print help message                      }";

bool isFileExist(const String fileName)
{
    ifstream file(fileName);
    return file.good();
}

template <typename T>
vector<int> sort_indexes(const vector<T>& v) {
    vector<int> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] > v[i2]; });

    return idx;
}

vector<int> kLargest(Mat data, int k)
{
    vector<int> ids = sort_indexes(vector<float>(data.begin<float>(), data.end<float>()));
    ids.resize(k);
    return ids;
}

int main(int argc, char** argv)
{
    // Process input arguments
    CommandLineParser parser(argc, argv, cmdOptions);
    parser.about(cmdAbout);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // Check all parameters
    if (!parser.has("image") || !parser.has("width") || !parser.has("heigth") || !parser.has("model_path") || !parser.has("config_path") || !parser.has("label_path"))
    {
        cout << "Not all required parameters provided. Required parameters are: image, width, heigth, model_path, config_path, label_path.";
        return -1;
    }

    // Load image and init parameters
    String imgName(parser.get<String>("image"));
    int imgWidth(parser.get<int>("width"));
    int imgHeigth(parser.get<int>("heigth"));
    String modelPath(parser.get<String>("model_path"));
    String configPath(parser.get<String>("config_path"));
    String labelsPath(parser.get<String>("label_path"));
    Scalar mean(parser.get<Scalar>("mean"));
    bool swap(parser.get<bool>("swap"));
    Scalar boundaries(parser.get<Scalar>("boundaries"));
    int top(parser.get<int>("top"));

    // Check all files
    if (!isFileExist(imgName))
    {
        cout << "Unable to open image file";
        return -1;
    }
    if (!isFileExist(modelPath))
    {
        cout << "Unable to open model file";
        return -1;
    }
    if (!isFileExist(configPath))
    {
        cout << "Unable to open config file";
        return -1;
    }
    if (!isFileExist(labelsPath))
    {
        cout << "Unable to open labels file";
        return -1;
    }

    Mat img = cv::imread(imgName);

    // Image cropping
    if (boundaries.val[0] >= 0 && boundaries.val[1] >= 0 &&
        img.size().width > boundaries.val[2] && boundaries.val[2] > boundaries.val[0] &&
        img.size().height > boundaries.val[3] && boundaries.val[3] > boundaries.val[1])
    {
        Rect newROI(boundaries.val[0], boundaries.val[1], boundaries.val[2] - boundaries.val[0], boundaries.val[3] - boundaries.val[1]);
        img = img(newROI);
    }

    // Image classification
    Classificator* classificator = new DnnClassificator(modelPath, configPath, labelsPath, imgWidth, imgHeigth, mean, swap);
    Mat result = classificator->Classify(img);

    // Result
    if (top >= 0)
    {
        result = result.reshape(1, 1);
        vector<int> ids = kLargest(result, top);

        cout << "The image is:" << endl;
        for (int i = 0; i < top; i++)
        {
            cout << "-\"" << classificator->GetLabels()[ids.at(i)] << "\" with the confidence of " << result.at<float>(ids.at(i)) * 100 << "%." << endl;
        }
    }
    else
    {
        Point classIdPoint;
        double confidence;
        minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
        cout << "The image is \"" << classificator->GetLabels()[classIdPoint.x] << "\" with the confidence of " << confidence * 100 << "%.";
    }

    return 0;
}
