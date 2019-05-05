#include "functions2.h"

void refitEIEModel()
{
    std::string faceDbPath = "./data/face_database";
    std::string eieModelPath = "./data/ELM_Models";
    
    ELM_IN_ELM_Model eieModel;
    
    int nModels = 10;//超参1:elm模型数目
    eieModel.setInitPara(nModels,eieModelPath);
    eieModel.loadStandardFaceDataset(faceDbPath,1,50,50);//超参2:resize大小
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,200);//超参3:elm隐藏层节点数
    eieModel.fitSubModels();
    eieModel.fitMainModel();
    eieModel.save();
}

void markImg(cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    img.at<cv::Vec3b>(0,0)[0] = 101;
    img.at<cv::Vec3b>(0,0)[1] = 100;
    img.at<cv::Vec3b>(0,0)[2] = 101;
    img.at<cv::Vec3b>(r,0)[0] = 100;
    img.at<cv::Vec3b>(r,0)[1] = 101;
    img.at<cv::Vec3b>(r,0)[2] = 100;
    img.at<cv::Vec3b>(0,c)[0] = 101;
    img.at<cv::Vec3b>(0,c)[1] = 100;
    img.at<cv::Vec3b>(0,c)[2] = 101;
    img.at<cv::Vec3b>(r,c)[0] = 100;
    img.at<cv::Vec3b>(r,c)[1] = 101;
    img.at<cv::Vec3b>(r,c)[2] = 100;
}

bool isMarkedImg(const cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    std::string key;
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[2]));
    
    //std::cout<<key<<std::endl;
    
    if(key == "101100101100101100101100101100101100")
        return true;
    else
        return false;
}
