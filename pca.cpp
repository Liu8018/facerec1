#include "pca.h"

PCA_Face pcaFace;

PCA_Face::PCA_Face()
{
    
}

cv::Mat PCA_Face::norm_0_255(const cv::Mat& src)
{
    cv::Mat dst;
    switch(src.channels())
        {
    case 1:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
        }
    return dst;
}

cv::Mat PCA_Face::asRowMatrix(const std::vector<cv::Mat>& src, int rtype, double alpha, double beta)
{
    //样本数量
    size_t n = src.size();
    //如果没有样本，返回空矩阵
    if(n == 0)
        return cv::Mat();
    //样本的维数
    size_t d = src[0].total();

    cv::Mat data(n, d, rtype);
    //拷贝数据
    for(int i = 0; i < n; i++)
        {
        cv::Mat xi = data.row(i);
        //转化为1行，n列的格式
        if(src[i].isContinuous())
            {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
            } else {
                src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
            }
        }
    return data;
}

void vectors2mat(const std::vector<std::vector<float> > &vecs, cv::Mat &mat)
{
    if(vecs.empty())
        return;
    
    int rows = vecs.size();
    int cols = vecs[0].size();
    mat.create(cv::Size(cols,rows),CV_32F);
    
    for(int i=0;i<vecs.size();i++)
    {
        for(int j=0;j<vecs[i].size();j++)
        {
            mat.at<float>(i,j) = vecs[i][j];
        }
    }
}

void getLbpData(const std::vector<cv::Mat> &faces, cv::Mat &data)
{
    std::vector<std::vector<float>> feats(faces.size());
    for(int i=0;i<faces.size();i++)
        alignment.extract_highdim_lbp_features(faces[i],feats[i]);
    
    vectors2mat(feats,data);
}

#define USE_HIGHDIM_LBP 0

void PCA_Face::calc_face(std::vector<cv::Mat> &faces)
{
    if(!pca.eigenvalues.empty())
        return;
    
    cv::Mat data;
    if(USE_HIGHDIM_LBP)
        getLbpData(faces,data);
    else
        data = asRowMatrix(faces, CV_32F);
    
    pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.99);
}

void PCA_Face::reduceDim_face(const std::vector<cv::Mat> &faceImgs, cv::Mat &output)
{
    cv::Mat data;
    if(USE_HIGHDIM_LBP)
        getLbpData(faceImgs,data);
    else
        data = asRowMatrix(faceImgs, CV_32F);
    
    pca.project(data,output);
}

void PCA_Face::reduceDim_face(const cv::Mat &faceImg, cv::Mat &output)
{
    std::vector<cv::Mat> tmpImgs;
    tmpImgs.push_back(faceImg);
    
    reduceDim_face(tmpImgs,output);
}

void PCA_Face::calc_feat(std::vector<std::vector<float>> &feats)
{
    if(!pca.eigenvalues.empty())
        return;
    
    cv::Mat data;
    vectors2mat(feats,data);
    
    pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.99);
}

void PCA_Face::reduceDim_feat(const std::vector<std::vector<float> > &feats, cv::Mat &output)
{
    cv::Mat data;
    vectors2mat(feats,data);
    
    pca.project(data,output);
}

void PCA_Face::reduceDim_feat(const std::vector<float> &feats, cv::Mat &output)
{
    std::vector<std::vector<float>> tmpFeats;
    tmpFeats.push_back(feats);
    
    reduceDim_feat(tmpFeats,output);
}

void PCA_Face::write(std::string path)
{
    cv::FileStorage fswrite(path,cv::FileStorage::WRITE);
    pca.write(fswrite);
}

void PCA_Face::read(std::string path)
{
    if(!pca.eigenvalues.empty())
        return;
    
    cv::FileStorage fsread(path,cv::FileStorage::READ);
    pca.read(fsread.root());
}
