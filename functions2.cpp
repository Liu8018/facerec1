#include "functions2.h"

void refitEIEModel()
{
    std::string faceDbPath = "./data/face_database";
    std::string eieModelPath = "./data/ELM_Models";
    
    ELM_IN_ELM_Model eieModel;
    
    int nModels = 10;//超参1:elm模型数目
    eieModel.setInitPara(nModels,eieModelPath);
    eieModel.loadStandardFaceDataset(faceDbPath,1,50,50);//超参2:resize大小
    for(int i=0;i<nModels;i++)
        eieModel.setSubModelHiddenNodes(i,100);//超参3:elm隐藏层节点数
    eieModel.fitSubModels();
    eieModel.fitMainModel();
    eieModel.save();
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

void getFileByName(std::string path, std::vector<cv::Mat> &imgs)
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
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0) 
            continue; 
		else if(ptr->d_type == 8) //file
		{
			std::string fn(ptr->d_name);
			std::string p = path + fn;
            
            imgs.push_back(cv::imread(p,0));
		}
    }
    
    closedir(dir);
}
