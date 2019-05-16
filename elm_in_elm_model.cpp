#include "elm_in_elm_model.h"

ELM_IN_ELM_Model::ELM_IN_ELM_Model()
{
    
}
ELM_IN_ELM_Model::ELM_IN_ELM_Model(const int n_models, const std::string modelDir)
{
    m_n_models = n_models;
    m_subModelHiddenNodes.resize(m_n_models);
    
    for(int i=0;i<m_n_models;i++)
        m_subModelHiddenNodes[i] = -1;
    
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
}

void ELM_IN_ELM_Model::setInitPara(const int n_models, const std::string modelDir)
{
    m_n_models = n_models;
    m_subModelHiddenNodes.resize(m_n_models);
    
    for(int i=0;i<m_n_models;i++)
        m_subModelHiddenNodes[i] = -1;
    
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
}

void ELM_IN_ELM_Model::setSubModelHiddenNodes(const int modelId, const int n_nodes)
{
    if(modelId == -1)
    {
        for(int i=0;i<m_n_models;i++)
            m_subModelHiddenNodes[i] = n_nodes;
    }
    else
        m_subModelHiddenNodes[modelId] = n_nodes;
}

void ELM_IN_ELM_Model::loadStandardDataset(const std::string path, const float trainSampleRatio, 
                                           const int resizeWidth, const int resizeHeight, const int channels, bool shuffle)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = channels;
    
    inputImgsFrom(path,m_label_string,m_trainImgs,
                  m_testImgs,m_trainLabelBins,m_testLabelBins,
                  trainSampleRatio,m_channels,shuffle);
    m_Q = m_trainImgs.size();
}

void ELM_IN_ELM_Model::loadStandardFaceDataset(const std::string path, const float trainSampleRatio, 
                                               const int resizeWidth, const int resizeHeight, bool shuffle)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = 1;
    
    inputImgsFrom(path,m_label_string,m_trainImgs,
                  m_testImgs,m_trainLabelBins,m_testLabelBins,
                  trainSampleRatio,m_channels,shuffle);
    
    for(int i=0;i<m_trainImgs.size();i++)
    {
        cv::resize(m_trainImgs[i],m_trainImgs[i],cv::Size(m_width,m_height));
        equalizeIntensity(m_trainImgs[i]);
    }
    for(int i=0;i<m_testImgs.size();i++)
    {
        cv::resize(m_testImgs[i],m_testImgs[i],cv::Size(m_width,m_height));
        equalizeIntensity(m_testImgs[i]);
    }
    
    /*
    getAverageImg(m_trainImgs,m_averageFace);
    minusAverage(m_averageFace,m_trainImgs);
    minusAverage(m_averageFace,m_testImgs);
    */
    
    m_C = m_label_string.size();
    m_Q = m_trainImgs.size();
    
    /*
    //提取输入图像的lbp特征
    for(int i=0;i<m_trainImgs.size();i++)
    {
        cv::Mat lbp;
        LBP81(m_trainImgs[i],lbp);
        lbp.copyTo(m_trainImgs[i]);
    }
    for(int i=0;i<m_testImgs.size();i++)
    {
        cv::Mat lbp;
        LBP81(m_testImgs[i],lbp);
        lbp.copyTo(m_testImgs[i]);
    }
    */
}

void ELM_IN_ELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle)
{
    loadMnistData_csv(path,trainSampleRatio,
                      m_trainImgs,m_testImgs,m_trainLabelBins,m_testLabelBins,shuffle);
    
    m_Q = m_trainImgs.size();
    
    m_width = 28;
    m_height = 28;
    m_channels = 1;
}

void ELM_IN_ELM_Model::fitSubModels(int batchSize, bool validating, bool verbose)
{
    m_subModelToTrain.inputData_2d(m_trainImgs,m_trainLabelBins,m_width,m_height,m_channels);
    m_subModelToTrain.inputData_2d_test(m_testImgs,m_testLabelBins);
    
    int randomState = (unsigned)time(NULL);
    
    //训练子模型
    for(int i=0;i<m_n_models;i++)
    {
        if(m_subModelHiddenNodes[i] != -1)
            m_subModelToTrain.setHiddenNodes(m_subModelHiddenNodes[i]);
        m_subModelToTrain.setRandomState(randomState++);
        m_subModelToTrain.fit(batchSize, validating,verbose);
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
        
        m_subModelToTrain.clear();
    }
    
    pcaFace.write("./data/pca/pcaFace.xml");
}

void ELM_IN_ELM_Model::fitMainModel(int batchSize, bool validating, bool verbose)
{
    std::cout<<"【elm-in-elm训练开始】--------------------------------------"<<std::endl;
    
    if(m_trainImgs.empty())
        return;
    
    //载入子模型
    if(m_subModels.empty())
    {
        m_subModels.resize(m_n_models);
        for(int i=0;i<m_n_models;i++)
            m_subModels[i].load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                              m_modelPath+"subK"+std::to_string(i)+".xml");
    }
    
    //为H和T分配空间
    int M = m_n_models;
    if(batchSize==-1)
        batchSize = m_trainImgs.size();
    m_C = m_label_string.size();
    cv::Mat H(cv::Size(M*m_C,batchSize),CV_32F);
    cv::Mat batchTarget(cv::Size(m_C,batchSize),CV_32F);
    
    //转化标签 bool->Mat
    cv::Mat allTarget;
    label2target(m_trainLabelBins,allTarget);
    
    //输出矩阵大小信息
    if(verbose)
    {
        std::cout<<"Q: "<<m_Q<<std::endl
                 <<"batchSize: "<<batchSize<<std::endl
                 <<"M: "<<M<<std::endl
                 <<"C: "<<m_C<<std::endl;
    }
    
    //m_K的初始化
    if(m_K.empty())
    {
        m_K.create(cv::Size(M*m_C,M*m_C),CV_32F);
        m_K = cv::Scalar(0);
    }
    
    //m_F的初始化
    if(m_F.empty())
    {
        m_F.create(cv::Size(m_C,M*m_C),CV_32F);
        m_F = cv::Scalar(0);
    }
    
    int trainedRatio = 0;
    for(int i=0;i+batchSize<=m_Q;i+=batchSize)
    {
        std::vector<cv::Mat> batchTrainImgs(m_trainImgs.begin()+i,m_trainImgs.begin()+i+batchSize);

        //为H和batchTarget赋值
        for(int m=0;m<m_n_models;m++)
        {
            cv::Mat ROI = H(cv::Range(0,batchSize),cv::Range(m*m_C,(m+1)*m_C));
            m_subModels[m].batchQuery(batchTrainImgs,ROI);
        }
        batchTarget = allTarget(cv::Range(i,i+batchSize),cv::Range(0,m_C));

        //迭代更新K
        m_K = m_K + H.t() * H;
        
        //迭代更新F
        m_F = m_F + m_K.inv(1) * H.t() * (batchTarget - H*m_F);

        //输出信息
        if(verbose)
        {
            int ratio = (i+batchSize)/(float)m_Q*100;
            if( ratio - trainedRatio >= 1)
            {
                trainedRatio = ratio;
                
                //计算在该批次训练数据上的准确率
                cv::Mat output = H * m_F;
                float score = calcScore(output,batchTarget);
                std::cout<<"Score on batch training data:"<<score<<std::endl;
                
                //计算在测试数据上的准确率
                if(validating && m_testImgs.size()>0)
                    validate();
            }
        }
    }
    
    std::cout<<"【elm-in-elm训练结束】--------------------------------------"<<std::endl;
    
/*std::cout<<"T:"<<T.size<<"\n"<<T<<std::endl;
std::cout<<"H:"<<H.size<<"\n"<<H<<std::endl;
std::cout<<"F:"<<m_F.size<<"\n"<<m_F<<std::endl;
std::cout<<"H*F:"<<realOutput.size<<"\n"<<H*m_F<<std::endl;
*/
}

float ELM_IN_ELM_Model::validate()
{
    int M = m_n_models;
    
    cv::Mat H_test(cv::Size(M*m_C,m_testImgs.size()),CV_32F);
    cv::Mat T_test(cv::Size(m_C,m_testImgs.size()),CV_32F);
    
    //给H_test和T_test赋值
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H_test(cv::Range(0,m_testImgs.size()),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].batchQuery(m_testImgs,ROI);
    }
    label2target(m_testLabelBins,T_test);
    
    //计算
    cv::Mat output = H_test * m_F;
    float finalScore_test = calcScore(output,T_test);
    
    std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
    return finalScore_test;
}

void ELM_IN_ELM_Model::save()
{
    if(m_F.empty())
        return;
    
    cv::FileStorage fswrite(m_modelPath+"mainModel.xml",cv::FileStorage::WRITE);
    
    fswrite<<"n_models"<<m_n_models;
    fswrite<<"subModelPath"<<m_modelPath;
    fswrite<<"width"<<m_width;
    fswrite<<"height"<<m_height;
    fswrite<<"channels"<<m_channels;
    fswrite<<"C"<<m_C;
    fswrite<<"F"<<m_F;
    fswrite<<"label_string"<<m_label_string;
    pcaFace.write("./data/pca/pcaFace.xml");
    
    fswrite.release();
    
    cv::FileStorage K_fswrite(m_modelPath+"mainK.xml",cv::FileStorage::WRITE);
    K_fswrite<<"K"<<m_K;
    K_fswrite.release();
    
    if(!m_allFeats.empty())
        writeFeats();
}

void ELM_IN_ELM_Model::load(std::string modelDir)
{
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
    
    cv::FileStorage fsread(m_modelPath+"mainModel.xml",cv::FileStorage::READ);

    fsread["n_models"]>>m_n_models;
    fsread["subModelPath"]>>m_modelPath;
    fsread["channels"]>>m_channels;
    fsread["width"]>>m_width;
    fsread["height"]>>m_height;
    fsread["C"]>>m_C;
    fsread["F"]>>m_F;
    fsread["label_string"]>>m_label_string;
    
    pcaFace.read("./data/pca/pcaFace.xml");

    fsread.release();
    
    cv::FileStorage K_fsread(m_modelPath+"mainK.xml",cv::FileStorage::READ);
    K_fsread["K"]>>m_K;
    K_fsread.release();

    //加载子模型
    m_subModels.resize(m_n_models);
    for(int m=0;m<m_n_models;m++)
    {
        m_subModels[m].load(m_modelPath+"subModel"+std::to_string(m)+".xml",
                            m_modelPath+"subK"+std::to_string(m)+".xml");
    }
    
    if(access("./data/face_database/lbpFeats.dat",F_OK) != -1)
        readFeats();
}

void ELM_IN_ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
    }
    
    cv::Mat output = H * m_F;
    //std::cout<<output<<std::endl;
    int maxId = getMaxId(output);
    
    label.assign(m_label_string[maxId]);
}

void ELM_IN_ELM_Model::queryFace(const cv::Mat &mat, std::string &label)
{
    cv::Mat gray;//,lbp;
    if(mat.channels() == 3)
        cv::cvtColor(mat,gray,cv::COLOR_BGR2GRAY);
    
    cv::resize(gray,gray,cv::Size(m_width,m_height));
    
    query(gray,label);
}

void myNormalize(cv::Mat &mat)
{
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
        {
            float val = mat.at<float>(i,j);
            
            if(val < 0)
                mat.at<float>(i,j) = 0.5/(1.0-val);
            else
                mat.at<float>(i,j) = 1.0 - 0.5/(1.0 + val);
        }
}

void getFileByName_s(std::string path, std::vector<cv::Mat> &imgs)
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

void ELM_IN_ELM_Model::query(const cv::Mat &mat, int n, std::map<float,std::string> &nameScores)
{
    if(m_F.empty())
        return;
    
    if(n>m_C)
        n = m_C;
    
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
    }
    
    cv::Mat output = H * m_F;
    
    std::vector<int> maxIds;
    getMaxNId(output,n,maxIds);
    
    //相似度计算
    cv::Mat om;
    pcaFace.reduceDim_face(mat,om);
    
    float a = cv::norm(om);
    
    for(int i=0;i<n;i++)
    {
        int id = maxIds[i];        
        std::string name = m_label_string[id];
        
        float similarity = -1;
        
        for(int j=0;j<m_allFeats[id].size();j++)
        {
            /*
            std::cout<<"om:\n"<<om<<std::endl;
            std::cout<<"prj:\n"<<prj<<std::endl;
            std::cout<<"om - prj:\n"<<om - prj<<std::endl;
            */
            
            float b = cv::norm(m_allFeats[id][j]);
            float c = cv::norm(om - m_allFeats[id][j]);
            
            float sim = (a*a+b*b-c*c)/(2*a*b);
            
            //std::cout<<name<<": "<<sim<<std::endl;
            
            if(similarity == -1)
                similarity = sim;
            else
            {
                if(sim < similarity)
                    similarity = sim;
            }
        }
        
        //赋值并自动排序
        nameScores.insert(std::pair<float,std::string>(similarity,name));
    }
    
}

void ELM_IN_ELM_Model::calcFeats()
{
    m_allFeats.clear();
    
    for(int i=0;i<m_C;i++)
    {
        std::vector<cv::Mat> nameFeats;
        
        std::string name = m_label_string[i];
        
        std::vector<cv::Mat> dbImgs;
        getFileByName_s("./data/face_database/"+name,dbImgs);
        
        for(int j=0;j<dbImgs.size();j++)
        {
            cv::Mat prj;
            cv::resize(dbImgs[j],dbImgs[j],cv::Size(m_width,m_height));
            pcaFace.reduceDim_face(dbImgs[j],prj);
            
            nameFeats.push_back(prj);
        }
        
        m_allFeats.push_back(nameFeats);
    }
}
void ELM_IN_ELM_Model::writeFeats()
{
    cv::FileStorage fswrite("./data/face_database/lbpFeats.dat",cv::FileStorage::WRITE);
    
    fswrite<<"allFeats"<<m_allFeats;
    
    fswrite.release();
}

void ELM_IN_ELM_Model::readFeats()
{
    cv::FileStorage fsread("./data/face_database/lbpFeats.dat",cv::FileStorage::READ);
    
    fsread["allFeats"]>>m_allFeats;
    
    fsread.release();
}

void ELM_IN_ELM_Model::queryFace(const cv::Mat &mat, int n, std::map<float,std::string> &nameScores)
{
    /*
    cv::Mat gray;
    if(mat.channels() == 3)
        cv::cvtColor(mat,gray,cv::COLOR_BGR2GRAY);
    else
        mat.copyTo(gray);
    */
    
    cv::Mat mat2;
    cv::resize(mat,mat2,cv::Size(m_width,m_height));
    query(mat2,n,nameScores);
}

void ELM_IN_ELM_Model::clearTrainData()
{
    if(m_trainImgs.empty())
        return;
    
    
    m_subModelToTrain.clearTrainData();
    for(int i=0;i<m_subModels.size();i++)
        m_subModels[i].clearTrainData();
    
    if(!m_trainImgs.empty())
        m_trainImgs.clear();
    if(!m_trainLabelBins.empty())
        m_trainLabelBins.clear();
    if(!m_testImgs.empty())
        m_testImgs.clear();
    if(!m_testLabelBins.empty())
        m_testLabelBins.clear();
}

void ELM_IN_ELM_Model::trainNewImg(const cv::Mat &img, const std::string label)
{
    clearTrainData();
    m_trainImgs.push_back(img);
    std::vector<bool> labelBin(m_C,0);
    for(int i=0;i<m_label_string.size();i++)
        if(label == m_label_string[i])
        {
            labelBin[i] = 1;
            break;
        }
    m_trainLabelBins.push_back(labelBin);
    m_Q = 1;
    
    fitSubModels(-1,false,false);
    fitMainModel(-1,false,false);
}

void ELM_IN_ELM_Model::trainNewFace(const cv::Mat &img, const std::string label)
{
    cv::Mat gray;//,lbp;
    if(img.channels() == 3)
        cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    else
        img.copyTo(gray);
    
    cv::resize(gray,gray,cv::Size(m_width,m_height));
    
    trainNewImg(gray,label);
}
