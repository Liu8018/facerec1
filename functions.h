#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "FaceAlignment.h"
#include "FaceRecognition.h"

//dlib人脸识别用到的
void getFiles(std::string path, std::map<std::string, std::string> &files);
void updatedb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib);

#endif // FUNCTIONS_H
