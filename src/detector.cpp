#include "detector.h"

DnnDetector::DnnDetector(string pthModel, string pthConfig, string pthLabels, int inputWidth, int inputHeight,
    Scalar myMean, int mySwapRB) {
   
    model = pthModel;
    crop = false;


}

vector<DetectedObject> DnnDetector::Detect(Mat image)
{ 

    string arLabels[] = {
    "background",
        "eroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    };

    Size spatialSize = Size(width, height);
    blobFromImage(image, blob, scale, spatialSize, mean, swapRB, crop, ddepth);
    net.setInput(blob);
    Mat prob = net.forward();
    
    vector<DetectedObject> res;

   
   Mat mx = prob.reshape(1, 1);
   int cols = mx.cols;
   int rows = mx.cols/7;

   mx = mx.reshape(1, rows);

   cout << mx;
   
   DetectedObject tmp;
   for (int i = 0; i < rows; i++) {
       
       tmp.uuid = mx.at<float>(i, 1);
       tmp.score = mx.at<float>(i, 2);
       tmp.Left = mx.at<float>(i,3)*image.cols;
       tmp.Bottom = mx.at<float>(i, 6)*image.rows;
       tmp.Right = mx.at<float>(i, 4)*image.cols;
       tmp.Top = mx.at<float>(i, 5)*image.rows;
       tmp.classname = arLabels[tmp.uuid];

       if (tmp.score > 0.9) {
           res.push_back(tmp);
       }
   }
   
   
   

   // for(int i = 0; i < )
  // cout << res1;
    

    return res;
}
