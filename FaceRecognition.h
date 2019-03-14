#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <map>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

#include "elm_in_elm_model.h"

//定义好一堆模板别名，以供后续方便使用
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<alevel1<alevel2<alevel3<alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

class FaceRecognition
{
public:
    FaceRecognition() {}
    FaceRecognition(std::string recMethod);
   
    //dlib人脸识别    
    bool recognize(const cv::Mat &img, const dlib::full_object_detection &shape, std::string &name);
    
    //ELM_IN_ELM人脸识别
    bool recognize(const cv::Mat &faceImg, std::string &name);
    bool recognize(const cv::Mat &faceImg, int n, std::vector<std::string> &names);
    
    void getDescriptor(const cv::Mat &src, const dlib::full_object_detection &shape, dlib::matrix<float,0,1> &faceDescriptor);
    
    void init_updatedb();
    void init_loadDb();
    
    void init_updateEIEdb();
    void init_loadEIEdb();
    
    void updateEIEdb(const cv::Mat &img, const std::string label);
    
    std::string method;
    
private:
    std::string m_modelPath;
    
    anet_type m_net;
    
    ELM_IN_ELM_Model m_eieModel;
    
    //人脸描述符库
    std::map<dlib::matrix<float,0,1>, std::string> m_faceDescriptorsLib;
};

#endif // FACERECOGNITION_H
