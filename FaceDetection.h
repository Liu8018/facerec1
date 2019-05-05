#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>


class FaceDetection
{
public:
    FaceDetection();
    
    void detect(const cv::Mat &src, std::vector<cv::Rect> &faceRects);
    
private:
    dlib::frontal_face_detector m_ffd;
};

#endif // FACEDETECTOR_H
