#include "functions.h"

void getFiles(std::string path, std::map<std::string, std::string> &files)
{
	DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == NULL)
	{
		std::cout<<"open the dir: "<< path <<" error!" <<std::endl;
		return;
	}
	
	while((ptr=readdir(dir)) !=NULL )
	{
		///current dir OR parrent dir 
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 8) //file
		{
			std::string fn(ptr->d_name);
            
            if(fn.substr(fn.length()-3,fn.length()-1) == "dat")
                continue;
            
            std::string className = path;
            className.pop_back();
            className = className.substr(className.find_last_of("/")+1,className.length()-1);
            
            //只有文件名中带有人名(上级文件夹名)的图片，才会被加入
            //if(fn.find(className) == std::string::npos)
            //    continue;
            
			std::string p = path + fn;
			files.insert(std::pair<std::string, std::string>(p, className));
		}
		else if(ptr->d_type == 10)    ///link file
		{}
		else if(ptr->d_type == 4)    ///dir
		{
            std::string p = path + std::string(ptr->d_name);
            getFiles(p,files);
        }
	}
	
	closedir(dir);
	return ;
}

void updateResnetDb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib)
{
    std::string facedbPath = "./data/face_database";
    
    FaceAlignment alignment;
    FaceRecognition recognition("resnet");
    
    std::map<std::string, std::string> files;
    getFiles(facedbPath, files);
    
    for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++  )
    {
        std::cout << "name:" << it->second << "	filepath:" <<it->first<<std::endl;
        
        //转为dlib图像
        cv::Mat frame = cv::imread(it->first);
        if(frame.empty())
            continue;
        cv::Mat src;
        cv::cvtColor(frame, src, cv::COLOR_BGR2GRAY);
        dlib::array2d<dlib::bgr_pixel> dimg;
        dlib::assign_image(dimg, dlib::cv_image<uchar>(src)); 
        
        //得到shape
        dlib::full_object_detection shape;
        alignment.getShape(frame,cv::Rect(0,0,frame.cols,frame.rows),shape);

        //计算描述子
        dlib::matrix<float,0,1> faceDescriptor;
        recognition.getDescriptor(frame,shape,faceDescriptor);
        
        faceDescriptorsLib.insert(std::pair<dlib::matrix<float,0,1>, std::string>(faceDescriptor, it->second));
    }
    
    dlib::serialize("./data/face_database/faceDescriptors.dat") << faceDescriptorsLib;
}

void refitEIEModel()
{
    std::string faceDbPath = "./data/face_database";
    std::string eieModelPath = "./data/ELM_Models";
    
    ELM_IN_ELM_Model eieModel;
    
    int nModels = 10;
    eieModel.setInitPara(nModels,eieModelPath);
    eieModel.loadStandardFaceDataset(faceDbPath,1,50,50);
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,100);
    eieModel.fitSubModels();
    eieModel.fitMainModel();
    eieModel.save();
}

void handleFaceDb(int method)
{
    if(method == 1)
    {
        std::map<std::string, std::string> files;
        getFiles("./data/face_database",files);
        
        //人脸检测初始化
        FaceDetection detection;
        //人脸对齐初始化
        FaceAlignment alignment;
        
        //对库中图像进行人脸检测并裁剪、对齐
        for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++  )
        {
            cv::Mat image = cv::imread(it->first);
            
            if(isMarkedImg(image))
                continue;
            
            std::cout <<"handling file:" <<it->first<<std::endl;
            
            std::vector<cv::Rect> objects;
            detection.detect(image, objects);
            
            cv::Rect faceRect;
            if(objects.empty())
                faceRect = cv::Rect(0,0,image.cols,image.rows);
            else
                faceRect = objects[0];
            
            //std::cout<<"image.size: "<<image.size<<std::endl;
            //std::cout<<"faceRect: "<<faceRect<<std::endl;
            
            //对齐
            dlib::full_object_detection shape;
            alignment.getShape(image,faceRect,shape);
            cv::Mat resultImg;
            alignment.alignFace(image,faceRect,resultImg);
            
            //cv::imshow("detect+alignment",resultImg);
            //cv::waitKey();
            
            //输出
            markImg(resultImg);
            std::string outFile = it->first;
            outFile = outFile.substr(0,outFile.find_last_of("."));
            outFile += ".png";//jpg编码存取数据不一致，必须转成png格式
            remove(it->first.data());
            cv::imwrite(outFile,resultImg);
        }
        
        //重新训练elm-in-elm模型
        refitEIEModel();
    }
    
    if(method == 2)
    {
        //重新用resnet模型提取特征库
        std::cout<<"updating resnet"<<std::endl;
        std::map<dlib::matrix<float,0,1>, std::string> faceDescriptorsLib;
        updateResnetDb(faceDescriptorsLib);
    }
}

void markImg(cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    img.at<cv::Vec3b>(0,0)[0] = 101;
    img.at<cv::Vec3b>(0,0)[1] = 100;
    img.at<cv::Vec3b>(0,0)[2] = 101;
    img.at<cv::Vec3b>(r,0)[0] = 100;
    img.at<cv::Vec3b>(r,0)[1] = 101;
    img.at<cv::Vec3b>(r,0)[2] = 100;
    img.at<cv::Vec3b>(0,c)[0] = 101;
    img.at<cv::Vec3b>(0,c)[1] = 100;
    img.at<cv::Vec3b>(0,c)[2] = 101;
    img.at<cv::Vec3b>(r,c)[0] = 100;
    img.at<cv::Vec3b>(r,c)[1] = 101;
    img.at<cv::Vec3b>(r,c)[2] = 100;
}

bool isMarkedImg(const cv::Mat &img)
{
    int c = img.cols-1;
    int r = img.rows-1;
    
    std::string key;
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,0)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(0,c)[2]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[0]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[1]));
    key.append(std::to_string(img.at<cv::Vec3b>(r,c)[2]));
    
    //std::cout<<key<<std::endl;
    
    if(key == "101100101100101100101100101100101100")
        return true;
    else
        return false;
}
