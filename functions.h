#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "FaceAlignment.h"
#include "FaceRecognition.h"

void updatedb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib);

#endif // FUNCTIONS_H
