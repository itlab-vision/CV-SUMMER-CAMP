#include "classificator.h"
#include <opencv2/dnn.hpp>



DnnClassificator::DnnClassificator(string pthModel, string pthConfig, string pthLabels, int inputWidth, int inputHeight, Scalar myMean, int mySwapRB)
{
    model = pthModel;
    config = pthConfig;
    labels = pthLabels;
    width = inputWidth;
    height = inputHeight;
    mean = myMean;
    swapRB = mySwapRB;

    backendId = DNN_BACKEND_OPENCV;
    targetId = DNN_TARGET_CPU;
    net = readNet(model, config);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    scale = 1;
    ddepth = CV_32F;
    crop = false;
}

Mat DnnClassificator::Classify(Mat image)
{
    Size spatialSize = Size(width, height);
    blobFromImage(image, blob, scale, spatialSize, mean, swapRB, crop, ddepth);
    net.setInput(blob);

    return net.forward();
}
