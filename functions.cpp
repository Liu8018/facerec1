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
            if(fn.find(className) == std::string::npos)
                continue;
            
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

void getFiles2(std::string path, std::map<std::string, std::string> &files)
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
            
			std::string p = path + fn;
			files.insert(std::pair<std::string, std::string>(p, className));
		}
		else if(ptr->d_type == 10)    ///link file
		{}
		else if(ptr->d_type == 4)    ///dir
		{
            std::string p = path + std::string(ptr->d_name);
            getFiles2(p,files);
        }
	}
	
	closedir(dir);
	return ;
}

void updatedb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib)
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

void handleFaceDb()
{
    std::string faceDbPath = "./data/face_database";    
    std::string eieModelPath = "./data/ELM_Models";
    
    std::vector<std::string> label_string;
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    inputImgsFrom(faceDbPath,label_string,trainImgs,testImgs,trainLabelBins,testLabelBins,1,3);
    
    //人脸检测初始化
    SimdDetection detection;
    detection.Load("./data/cascade/haar_face_0.xml");
    //人脸对齐初始化
    FaceAlignment alignment;
    
    for(int i=0;i<trainImgs.size();i++)
    {
        cv::Size srcSize = trainImgs[i].size();
        detection.Init(srcSize,1.2,srcSize/5);
        SimdDetection::View image = trainImgs[i];
        SimdDetection::Objects objects;
        detection.Detect(image, objects);
        
        cv::Rect faceRect;
        if(objects.empty())
            faceRect = cv::Rect(0,0,trainImgs[i].cols,trainImgs[i].rows);
        else
            faceRect = objects[0].rect;
        
        //对齐
        dlib::full_object_detection shape;
        alignment.getShape(trainImgs[i],faceRect,shape);
        cv::Mat resultImg;
        alignment.alignFace(trainImgs[i],faceRect,resultImg);
        
        cv::cvtColor(resultImg,trainImgs[i],cv::COLOR_BGR2GRAY);
    }
    
    ELM_IN_ELM_Model eieModel;
    int nModels = 10;
    eieModel.setInitPara(nModels,eieModelPath);
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,100);
    eieModel.loadFaces(trainImgs,label_string,trainLabelBins,50,50);
    eieModel.fitSubModels();
    eieModel.fitMainModel();
    eieModel.save();
    
    /*
    ELM_IN_ELM_Model eieModel;
    int nModels = 10;
    eieModel.setInitPara(nModels,eieModelPath);
    eieModel.loadStandardFaceDataset(faceDbPath,1,50,50);
    eieModel.clearTrainData();
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,100);
    
    //人脸检测初始化
    SimdDetection detection;
    detection.Load("./data/cascade/haar_face_0.xml");
    //人脸对齐初始化
    FaceAlignment alignment;
    
    //对库中图像进行人脸检测并裁剪、对齐,再用来训练elm-in-elm模型
    std::cout<<"updating elm"<<std::endl;
    std::map<std::string, std::string> files;
    getFiles2(faceDbPath,files);
    for(std::map<std::string, std::string>::iterator it = files.begin(); it != files.end(); it++)
    {
        std::string fileName = it->first;
        cv::Mat frame = cv::imread(fileName);
        if(frame.empty())
            continue;
                
        cv::Size srcSize = frame.size();
        detection.Init(srcSize,1.2,srcSize/5);
        SimdDetection::View image = frame;
        SimdDetection::Objects objects;
        detection.Detect(image, objects);
        
        cv::Rect faceRect;
        if(objects.empty())
            faceRect = cv::Rect(0,0,frame.cols,frame.rows);
        else
            faceRect = objects[0].rect;
        
        //对齐
        dlib::full_object_detection shape;
        alignment.getShape(frame,faceRect,shape);
        cv::Mat resultImg;
        alignment.alignFace(frame,faceRect,resultImg);
        
        //训练
        eieModel.trainNewFace(resultImg,it->second);
    }
    */
    
    //重新用resnet模型提取特征库
    std::cout<<"updating resnet"<<std::endl;
    std::map<dlib::matrix<float,0,1>, std::string> faceDescriptorsLib;
    updatedb(faceDescriptorsLib);
}
