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

DnnClassificator::DnnClassificator(string path_model, string path_config, string path_labels, int inpw,
                                   int inph, bool sw, double   sc , Scalar m,int back, int targ)
{
    model = path_model;
    config = path_config;
    labels = path_labels;
    inputHeight = inph;
    inputWidth = inpw;
    swapRB = sw;
    scale = sc;
    mean  = m;
    backendId = back;
    targetId = targ;
    
    net = readNet(model, config);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    
    }
