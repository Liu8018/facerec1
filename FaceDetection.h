#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "params.h"
#include <opencv2/dnn.hpp>


class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &img, std::vector<cv::Rect> &boxes);
    
private:
    cv::dnn::Net m_net;
};

#endif // FACEDETECTOR_H
