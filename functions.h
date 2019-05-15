#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "FaceDetection.h"
#include "FaceAlignment.h"
#include "FaceRecognition.h"

#include "functions2.h"

//升级resnet数据库
void updateResnetDb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib);

//升级数据库
void handleFaceDb(int method);

#endif // FUNCTIONS_H
