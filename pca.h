#ifndef PCA_H
#define PCA_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

class PCA_Face
{
public:
    PCA_Face();
    
    void calc(std::vector<cv::Mat> &faces);
    
    cv::Mat norm_0_255(const cv::Mat& src);
    cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype, double alpha = 1, double beta = 0);
    
    void reduceDim(const std::vector<cv::Mat> &faceImgs, cv::Mat &outputData);
    void reduceDim(const cv::Mat &faceImg, cv::Mat &outputData);
    
    void read(std::string path);
    void write(std::string path);
    
    cv::PCA pca;
    
private:
};

#endif // PCA_H
