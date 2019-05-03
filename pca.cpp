#include "pca.h"

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

        if(src[i].empty()) 
            {
            std::string error_message = cv::format("Image number %d was empty, please check your input data.", i);
            //CV_Error(CV_StsBadArg, error_message);
            }
        // 确保数据能被reshape
        if(src[i].total() != d) 
            {
            std::string error_message = cv::format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            //CV_Error(CV_StsBadArg, error_message);
            }
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

void PCA_Face::calc(std::vector<cv::Mat> &faces)
{
    //if(!m_pca.eigenvalues.empty())
    //    return;
    
    cv::Mat data = asRowMatrix(faces, CV_32FC1);//Q*MN
    //std::cout<<data.size<<std::endl;
    m_pca = cv::PCA(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.99);
}

void PCA_Face::reduceDim(const std::vector<cv::Mat> &faceImgs, cv::Mat &outputData)
{
    cv::Mat data = asRowMatrix(faceImgs, CV_32FC1);
    m_pca.project(data,outputData);
}

void PCA_Face::write(std::string path)
{
    cv::FileStorage fswrite(path,cv::FileStorage::WRITE);
    m_pca.write(fswrite);
}

void PCA_Face::read(std::string path)
{
    cv::FileStorage fsread(path,cv::FileStorage::READ);
    m_pca.read(fsread.root());
}
