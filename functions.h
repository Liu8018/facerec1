#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "FaceDetection.h"
#include "FaceAlignment.h"
#include "FaceRecognition.h"

//获取标准文件夹格式下的文件
void getFiles(std::string path, std::map<std::string, std::string> &files);

//升级resnet数据库
void updateResnetDb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib);

//重新训练elm-in-elm模型
void refitEIEModel();

//升级数据库
void handleFaceDb(int method);

//给库中检测并裁剪过的人脸图片打上标记
void markImg(cv::Mat &img);

//检测图片是否已被打上标记
bool isMarkedImg(const cv::Mat &img);

#endif // FUNCTIONS_H
