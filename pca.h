#ifndef PCA_H
#define PCA_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include "FaceAlignment.h"

class PCA_Face
{
public:
    PCA_Face();
    
    void calc_face(std::vector<cv::Mat> &faces);
    void calc_feat(std::vector<std::vector<float> > &feats);
    
    cv::Mat norm_0_255(const cv::Mat& src);
    cv::Mat asRowMatrix(const std::vector<cv::Mat>& src, int rtype, double alpha = 1, double beta = 0);
    
    void reduceDim_face(const std::vector<cv::Mat> &faceImgs, cv::Mat &output);
    void reduceDim_face(const cv::Mat &faceImg, cv::Mat &output);
    void reduceDim_feat(const std::vector<std::vector<float>> &feats, cv::Mat &output);
    void reduceDim_feat(const std::vector<float> &feats, cv::Mat &output);
    
    void read(std::string path);
    void write(std::string path);
    
    cv::PCA pca;
    
private:
};

extern PCA_Face pcaFace;

#endif // PCA_H
