#include "functions.h"

void updateResnetDb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib)
{
    std::string facedbPath = "./data/face_database";
    
    FaceAlignment alignment;
    FaceRecognition recognition("resnet");
    
    std::map<std::string, std::string> files;
    getFiles(facedbPath, files);
    
    for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++  )
    {
        std::cout << "name:" << it->second << "	filepath:" <<it->first<<std::endl;
        
        //转为dlib图像
        cv::Mat frame = cv::imread(it->first);
        if(frame.empty())
            continue;
        cv::Mat src;
        cv::cvtColor(frame, src, cv::COLOR_BGR2GRAY);
        dlib::array2d<dlib::bgr_pixel> dimg;
        dlib::assign_image(dimg, dlib::cv_image<uchar>(src)); 
        
        //得到shape
        dlib::full_object_detection shape;
        alignment.getShape(frame,cv::Rect(0,0,frame.cols,frame.rows),shape);

        //计算描述子
        dlib::matrix<float,0,1> faceDescriptor;
        recognition.getDescriptor(frame,shape,faceDescriptor);
        
        faceDescriptorsLib.insert(std::pair<dlib::matrix<float,0,1>, std::string>(faceDescriptor, it->second));
    }
    
    dlib::serialize("./data/face_database/faceDescriptors.dat") << faceDescriptorsLib;
}

void handleFaceDb(int method)
{
    if(method == 1)
    {
        std::map<std::string, std::string> files;
        getFiles("./data/face_database",files);
        
        //人脸检测初始化
        FaceDetection detection;
        //人脸对齐初始化
        FaceAlignment alignment;
        
        //对库中图像进行人脸检测并裁剪、对齐
        for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++  )
        {
            cv::Mat image = cv::imread(it->first);
            
            if(isMarkedImg(image))
                continue;
            
            std::cout <<"handling file:" <<it->first<<std::endl;
            
            std::vector<cv::Rect> objects;
            detection.detect(image, objects);
            
            cv::Rect faceRect;
            if(objects.empty())
                faceRect = cv::Rect(0,0,image.cols-1,image.rows-1);
            else
                faceRect = objects[0];
            
            //std::cout<<"image.size: "<<image.size<<std::endl;
            //std::cout<<"faceRect: "<<faceRect<<std::endl;
            
            //对齐
            dlib::full_object_detection shape;
            alignment.getShape(image,faceRect,shape);
            cv::Mat resultImg;
            alignment.alignFace(image,faceRect,resultImg);
            
            //cv::imshow("detect+alignment",resultImg);
            //cv::waitKey();
            
            //输出
            markImg(resultImg);
            std::string outFile = it->first;
            outFile = outFile.substr(0,outFile.find_last_of("."));
            outFile += ".png";//jpg编码存取数据不一致，必须转成png格式
            remove(it->first.data());
            cv::imwrite(outFile,resultImg);
        }
        
        //重新训练elm-in-elm模型
        refitEIEModel();
    }
    
    if(method == 2)
    {
        //重新用resnet模型提取特征库
        std::cout<<"updating resnet"<<std::endl;
        std::map<dlib::matrix<float,0,1>, std::string> faceDescriptorsLib;
        updateResnetDb(faceDescriptorsLib);
    }
}


