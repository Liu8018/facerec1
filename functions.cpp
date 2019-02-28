#include "functions.h"

void getFiles(std::string path, std::map<std::string, std::string> &files)
{
	DIR *dir;
	struct dirent *ptr;

	if(path[path.length()-1] != '/')
		path = path + "/";

	if((dir = opendir(path.c_str())) == NULL)
	{
		std::cout<<"open the dir: "<< path <<"error!" <<std::endl;
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
			std::string name;
			name = fn.substr(0, fn.find_last_of("."));

			std::string p = path + std::string(ptr->d_name);
			files.insert(std::pair<std::string, std::string>(p, name));
		}
		else if(ptr->d_type == 10)    ///link file
		{}
		else if(ptr->d_type == 4)    ///dir
		{}
	}
	
	closedir(dir);
	return ;
}

void updatedb(std::map<dlib::matrix<float,0,1>, std::string> &faceDescriptorsLib)
{
    std::string facedbPath = "./data/face_database";
    
    FaceAlignment alignment;
    FaceRecognition recognition;
    
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
        cv::cvtColor(frame, src, CV_BGR2GRAY);
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
}
