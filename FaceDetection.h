#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "facedetectcnn.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define DETECT_BUFFER_SIZE 0x20000

class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &src, std::vector<cv::Rect> &faceRects);
    
private:
    int * pResults;
    unsigned char * pBuffer;
    
    int resizeWidth;
};

#endif // FACEDETECTOR_H
