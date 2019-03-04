#include "FaceRecognition.h"
#include "functions.h"

FaceRecognition::FaceRecognition()
{
    
}

void FaceRecognition::setMethod(std::string recMethod)
{
    method = recMethod;
    
    if(method == "resnet")
        dlib::deserialize("./data/facerec_model/dlib_face_recognition_resnet_model_v1.dat") >> m_net;
}

void dlibPoint2cvPoint(const dlib::full_object_detection &S, std::vector<cv::Point> &L)
{
    for(unsigned int i = 0; i<S.num_parts();++i)
        L.push_back(cv::Point(S.part(i).x(),S.part(i).y()));
}

void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
    int i, j;

    for(j=0; j<fs.size(); j++)
    {
        cv::Point p1, p2;

        for(i = 0; i<67; i++)
        {
            // 下巴到脸颊 0 ~ 16
            //左边眉毛 17 ~ 21
            //右边眉毛 21 ~ 26
            //鼻梁     27 ~ 30
            //鼻孔        31 ~ 35
            //左眼        36 ~ 41
            //右眼        42 ~ 47
            //嘴唇外圈  48 ~ 59
            //嘴唇内圈  59 ~ 67
            switch(i)
            {
                case 16:
                case 21:
                case 26:
                case 30:
                case 35:
                case 41:
                case 47:
                case 59:
                    i++;
                    break;
                default:
                    break;
            }

            p1.x = fs[j].part(i).x();
            p1.y = fs[j].part(i).y();
            p2.x = fs[j].part(i+1).x();
            p2.y = fs[j].part(i+1).y();
            cv::line(img, p1, p2, cv::Scalar(0,0,255), 1);
        }

    }
}

bool FaceRecognition::recognize(const cv::Mat &src, const dlib::full_object_detection &shape, std::string &name)
{
    std::vector<cv::Point> landmarks;
    dlibPoint2cvPoint(shape,landmarks);
    
    cv::Mat image = src.clone();
    std::vector<dlib::full_object_detection> shapes;
    shapes.push_back(shape);
    line_one_face_detections(image,shapes);
    cv::imshow("testImage",image);
        
    //提取描述子
    dlib::matrix<float,0,1> faceDescriptor;
    getDescriptor(src,shape,faceDescriptor);

    //遍历库，查找相似图像
    float minDistance=1;
    std::string nearestFaceName;
    for(std::map<dlib::matrix<float,0,1>, std::string>::iterator it=m_faceDescriptorsLib.begin(); it != m_faceDescriptorsLib.end(); it++ )
    {
        float distance = dlib::length(it->first - faceDescriptor);
        
        std::cout << "name: " << it->second << " distance: " << distance << std::endl;
        
        if(distance < minDistance)
        {
            minDistance = distance;
            nearestFaceName = it->second;
        }
    }
    std::cout<<"---"<<std::endl;
    
    if(minDistance < 0.40)
    {
        name = nearestFaceName;
        return true;
    }
    else
        return false;
}

void FaceRecognition::getDescriptor(const cv::Mat &src, const dlib::full_object_detection &shape, dlib::matrix<float,0,1> &faceDescriptor)
{
    //提取描述子
    cv::Mat img;
    cv::cvtColor(src, img, cv::COLOR_BGR2GRAY);
    dlib::array2d<dlib::bgr_pixel> dimg;
    dlib::assign_image(dimg, dlib::cv_image<uchar>(img));
            
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(dimg, dlib::get_face_chip_details(shape,150,0.25), face_chip);
    
    faceDescriptor = m_net(face_chip);
}

void FaceRecognition::init_updatedb()
{
    updatedb(m_faceDescriptorsLib);
}

void FaceRecognition::init_loadDb()
{
    dlib::deserialize("./data/face_database/faceDescriptors.dat") >> m_faceDescriptorsLib;
}

void FaceRecognition::init_updateEIEdb()
{
    m_eieModel = ELM_IN_ELM_Model();
    
    int nModels = 10;
    m_eieModel.setInitPara(nModels,"./data/ELM_Models");
    m_eieModel.loadStandardDataset("./data/face_database",1,50,50,1);
    for(int i=0;i<nModels;i++)
        m_eieModel.setSubModelHiddenNodes(i,100);
    m_eieModel.fitSubModels();
    m_eieModel.fitMainModel();
    m_eieModel.save();
    
    //释放内存空间
    m_eieModel.clearTrainData();
}

void FaceRecognition::updateEIEdb(const cv::Mat &img, const std::string label)
{
    m_eieModel.trainNewImg(img,label);
    m_eieModel.save();
}

void FaceRecognition::init_loadEIEdb()
{
    m_eieModel.load("./data/ELM_Models");
}

bool FaceRecognition::recognize(const cv::Mat &faceImg, std::string &name)
{
    cv::Mat faceImg_gray;
    if(faceImg.channels() == 3)
        cv::cvtColor(faceImg,faceImg_gray,cv::COLOR_BGR2GRAY);
    else
        faceImg_gray = faceImg;
    
    m_eieModel.query(faceImg_gray,name);
    
    return true;
}
