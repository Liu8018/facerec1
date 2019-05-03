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

void FaceAlignment::getShape(const cv::Mat &inputImg, const cv::Rect &faceRect, dlib::full_object_detection &shape)
{
    //转换opencv图像为dlib图像
    dlib::cv_image<dlib::rgb_pixel> cimg(inputImg);

    //提取脸部特征点(68个),存储在shape
    dlib::rectangle face_dlibRect;
    cvRect2dlibRect(faceRect,face_dlibRect);
    shape = m_shapePredictor(cimg,face_dlibRect);
}

void FaceAlignment::drawShape(cv::Mat &img, dlib::full_object_detection shape)
{
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //test:在两眼之间画线
    for(int i=0;i<landmarks.size();i++)
    {
        cv::circle(img,landmarks[i],3,cv::Scalar(0,255,0),-1);
        cv::putText(img,std::to_string(i),landmarks[i],1,0.8,cv::Scalar(255,0,0));
    }
}

void FaceAlignment::alignFace(const cv::Mat &inputImg, const cv::Rect &faceRect, cv::Mat &resultImg)
{
    
    dlib::full_object_detection shape;
    getShape(inputImg,faceRect,shape);
    
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    //根据两眼连线对齐人脸
    cv::Mat faceROI = inputImg(faceRect);
    cv::Point leye = landmarks[36];
    cv::Point reye = landmarks[45];
    cv::Point center = cv::Point(faceROI.cols/2,faceROI.rows/2);
    double dy = reye.y - leye.y; 
    double dx = reye.x - leye.x; 
    double angle = atan2(dy, dx) * 180.0 / CV_PI; 
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1); 
    cv::warpAffine(faceROI, resultImg, rotMat, faceROI.size());
    
    
    //resultImg = inputImg(faceRect);
    
    //cv::imshow("testResult",resultImg);
    //test
    /*cv::Mat image = inputImg.clone();
    cv::line(image,landmarks[45],landmarks[36],cv::Scalar(0,255,0),1);
    for(int i=0;i<landmarks.size();i++)
    {
        cv::circle(image,landmarks[i],3,cv::Scalar(255,0,0));
        cv::putText(image,std::to_string(i),landmarks[i],1,1,cv::Scalar(0,255,0));
    }
    cv::imshow("testImage",image);*/
}
