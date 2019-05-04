#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "FaceDetection.h"
#include "FaceAlignment.h"
#include "FaceRecognition.h"

#include "functions2.h"

//获取标准文件夹格式下的文件
void getFiles(std::string path, std::map<std::string, std::string> &files);

//升级resnet数据库
void updateResnetDb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib);

//升级数据库
void handleFaceDb(int method);

#endif // FUNCTIONS_H
