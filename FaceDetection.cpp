#include "FaceDetection.h"

FaceDetection::FaceDetection()
{
    m_ffd = dlib::get_frontal_face_detector();
}

void FaceDetection::detect(const cv::Mat &src, std::vector<cv::Rect> &faceRects)
{
    dlib::array2d<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(src));
    std::vector<dlib::rectangle> dets = m_ffd(img);
    
    if(!dets.empty())
        faceRects.push_back(cv::Rect(dets[0].left(),dets[0].top(),dets[0].width(),dets[0].height()));
}
