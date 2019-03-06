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
