#include "classificator.h"
Mat DnnClassificator::Classify(Mat image)
{
    Mat inputTensor;
    blobFromImage(image, inputTensor, scale, Size(inputWidth, inputHeight), mean, swapRB, false);
    
    net.setInput(inputTensor);
    Mat prob = net.forward();
    prob = prob.reshape(1, 1);
    return prob;
}
