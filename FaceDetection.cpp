#include "FaceDetection.h"

FaceDetection::FaceDetection()
{
    pResults = NULL; 
    pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        exit(0);
    }

    resizeWidth = 100;//最小84
    resizeRatio = -1;
}

void FaceDetection::detect(const cv::Mat &src, std::vector<cv::Rect> &faceRects)
{
    resizeRatio = resizeWidth/(float)src.cols;
    
    //std::cout<<"src.size:"<<src.size<<std::endl;
    cv::Mat image;
    cv::resize(src,image,cv::Size(),resizeRatio,resizeRatio);
    
    //std::cout<<"image.size:"<<image.size<<std::endl;
    
    pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
    
    faceRects.clear();
    for(int i = 0; i < (pResults ? *pResults : 0); i++)
    {
        short * p = ((short*)(pResults+1))+142*i;
        int x = p[0];
        int y = p[1];
        int w = p[2];
        int h = p[3];
        int confidence = p[4];
        //int angle = p[5];

        x /= resizeRatio;
        y /= resizeRatio;
        w /= resizeRatio;
        h /= resizeRatio;

        //printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x,y,w,h,confidence, angle);

        if(confidence > 90)
        {
            cv::Rect faceRect = cv::Rect(x, y, w, h);
            if(faceRect.br().x > src.cols-1)
                faceRect.width = src.cols-1-faceRect.x;
            if(faceRect.br().y > src.rows-1)
                faceRect.height = src.rows-1-faceRect.y;
            if(faceRect.x < 0)
                faceRect.x = 0;
            if(faceRect.y < 0)
                faceRect.y = 0;
            
            faceRects.push_back(faceRect);
        }
    }
}
