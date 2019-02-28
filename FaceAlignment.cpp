#include "FaceAlignment.h"

FaceAlignment::FaceAlignment()
{
    dlib::deserialize("./data/shape_predictor/shape_predictor_68_face_landmarks.dat") >> m_shapePredictor;
}

void FaceAlignment::cvRect2dlibRect(const cv::Rect &cvRec, dlib::rectangle &dlibRec)
{
    dlibRec = dlib::rectangle((long)cvRec.tl().x, (long)cvRec.tl().y, (long)cvRec.br().x - 1, (long)cvRec.br().y - 1);
}

void FaceAlignment::dlibPoint2cvPoint(const dlib::full_object_detection &S, std::vector<cv::Point> &L)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
        L.push_back(cv::Point(S.part(i).x(),S.part(i).y()));
}

/*
void FaceAlignment::process(const cv::Mat &inputImg, const cv::Rect &faceRect, cv::Mat &sFace)
{
    //转换opencv图像为dlib图像
    dlib::cv_image<dlib::rgb_pixel> cimg(inputImg);

    //提取脸部特征点(68个),存储在shape
    dlib::rectangle face_dlibRect;
    cvRect2dlibRect(faceRect,face_dlibRect);
    dlib::full_object_detection shape = m_shapePredictor(cimg,face_dlibRect);
    
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //test:在两眼之间画线
    cv::Mat image = inputImg.clone();
    cv::line(image,landmarks[45],landmarks[36],cv::Scalar(0,255,0),1);
    cv::imshow("testImage",image);
}*/

void FaceAlignment::getShape(const cv::Mat &inputImg, const cv::Rect &faceRect, dlib::full_object_detection &shape)
{
    //转换opencv图像为dlib图像
    dlib::cv_image<dlib::rgb_pixel> cimg(inputImg);

    //提取脸部特征点(68个),存储在shape
    dlib::rectangle face_dlibRect;
    cvRect2dlibRect(faceRect,face_dlibRect);
    shape = m_shapePredictor(cimg,face_dlibRect);
}
