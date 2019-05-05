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

void modifyROI(const cv::Size imgSize, cv::Rect &rect)
{
    if(rect.br().x > imgSize.width-1)
        rect.width = imgSize.width-1-rect.x;
    if(rect.br().y > imgSize.height-1)
        rect.height = imgSize.height-1-rect.y;
    if(rect.x < 0)
        rect.x = 0;
    if(rect.y < 0)
        rect.y = 0;
}

void modifyRectByFacePt(const dlib::full_object_detection &shape, cv::Rect &rect)
{
    std::vector<cv::Point> landmarks;
    for(unsigned int i = 0; i<shape.num_parts();++i)
        landmarks.push_back(cv::Point(shape.part(i).x(),shape.part(i).y()));
    
    int b = landmarks[8].y;
    
    int l1 = landmarks[0].x;
    int l2 = landmarks[1].x;
    int l3 = landmarks[17].x;
    int l = l1<l2?(l1<l3?l1:l3):(l2<l3?l2:l3);
    
    int r1 = landmarks[14].x;
    int r2 = landmarks[15].x;
    int r3 = landmarks[16].x;
    int r = r1<r2?(r1<r3?r1:r3):(r2<r3?r2:r3);
    
    int t1 = landmarks[19].y;
    int t2 = landmarks[24].y;
    int t = t1<t2?t1:t2;
    
    int refLen = (r-l)/20;
    
    rect.x = l-refLen;
    rect.y = t-2*refLen;
    rect.width = r-rect.x+refLen;
    rect.height = b-rect.y;
}

void FaceAlignment::alignFace(const cv::Mat &inputImg, cv::Rect &faceRect, cv::Mat &resultImg)
{
    /*
    //把矩形框扩大一点
    int refLen = faceRect.width/5;
    faceRect.x -= refLen;
    faceRect.width += 2*refLen;
    faceRect.y -= refLen;
    faceRect.height += 2*refLen;
    modifyROI(inputImg.size(),faceRect);
    */
    
    //获取特征点
    dlib::full_object_detection shape;
    getShape(inputImg,faceRect,shape);
    
    
    //test
    cv::Mat testImg = inputImg.clone();
    cv::rectangle(testImg,faceRect,cv::Scalar(255,0,0));
    drawShape(testImg,shape);
    cv::imshow("tmpImg",testImg);
    
    
    //shape转化为landmarks
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    /*
    //根据特征点更正矩形框，鼻尖对齐矩形框中心
    cv::Point nosePt = landmarks[30];
    faceRect.x = nosePt.x - faceRect.width/2;
    faceRect.y = nosePt.y - faceRect.height/2;
    modifyROI(inputImg.size(),faceRect);
    */
    
    //以两眼连线偏角绕矩形框中心旋转整幅图像
    cv::Point leye = landmarks[36];
    cv::Point reye = landmarks[45];
    double dy = reye.y - leye.y; 
    double dx = reye.x - leye.x; 
    double angle = atan2(dy, dx) * 180.0 / CV_PI; 
    cv::Point center = cv::Point(faceRect.x+faceRect.width/2,faceRect.y+faceRect.height/2);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1); 
    cv::Mat rotatedImg;
    cv::warpAffine(inputImg, rotatedImg, rotMat, inputImg.size());
    
    //取这时矩形框内的部分作为人脸
    dlib::full_object_detection shape2;
    getShape(rotatedImg,faceRect,shape2);
    //drawShape(rotatedImg,shape2);
    //modifyRectByFacePt(shape2,faceRect);
    //modifyROI(rotatedImg.size(),faceRect);
    resultImg = rotatedImg(faceRect).clone();
    
}
