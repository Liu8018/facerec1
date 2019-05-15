//与dlib无关的一些函数（dlib编译太耗时了，所以把与它无关的函数分开编译）
#ifndef FUNCTIONS2_H
#define FUNCTIONS2_H

#include "elm_in_elm_model.h"

//重新训练elm-in-elm模型
void refitEIEModel();

//给库中检测并裁剪过的人脸图片打上标记
void markImg(cv::Mat &img);

//检测图片是否已被打上标记
bool isMarkedImg(const cv::Mat &img);

//获取标准文件夹格式下的文件
void getFiles(std::string path, std::map<std::string, std::string> &files);
void getFileByName(std::string path, std::vector<cv::Mat> &imgs);



#endif // FUNCTIONS2_H
