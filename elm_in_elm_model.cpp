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

void ELM_IN_ELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle)
{
    loadMnistData_csv(path,trainSampleRatio,
                      m_trainImgs,m_testImgs,m_trainLabelBins,m_testLabelBins,shuffle);
    
    m_Q = m_trainImgs.size();
    
    m_width = 28;
    m_height = 28;
    m_channels = 1;
}

void ELM_IN_ELM_Model::fitSubModels(int batchSize, bool validating)
{
    if(m_subModels.empty())
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
            m_subModelToTrain.fit(batchSize, validating);
            m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                                   m_modelPath+"subK"+std::to_string(i)+".xml");
            
            m_subModelToTrain.clear();
        }
    }
    else
    {
        for(int i=0;i<m_n_models;i++)
        {
            m_subModels[i].inputData_2d(m_trainImgs,m_trainLabelBins,m_width,m_height,m_channels);
            m_subModels[i].inputData_2d_test(m_testImgs,m_testLabelBins);
            
            m_subModels[i].fit(batchSize, validating);
            m_subModels[i].save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                                   m_modelPath+"subK"+std::to_string(i)+".xml");
        }
    }
}

void ELM_IN_ELM_Model::fitMainModel(int batchSize, bool validating)
{
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
    std::cout<<"Q: "<<m_Q<<std::endl
             <<"batchSize: "<<batchSize<<std::endl
             <<"M: "<<M<<std::endl
             <<"C: "<<m_C<<std::endl;
    
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
        int ratio = (i+batchSize)/(float)m_Q*100;
        if( ratio - trainedRatio >= 1)
        {
            trainedRatio = ratio;
            
            //输出训练进度
            std::cout<<"Trained "<<trainedRatio<<"%"<<
                       "----------------------------------------"<<std::endl;
            
            //计算在该批次训练数据上的准确率
            cv::Mat output = H * m_F;
            float score = calcScore(output,batchTarget);
            std::cout<<"Score on batch training data:"<<score<<std::endl;
            
            //计算在测试数据上的准确率
            if(validating && m_testImgs.size()>0)
                validate();
        }
    }
/*std::cout<<"T:"<<T.size<<"\n"<<T<<std::endl;
std::cout<<"H:"<<H.size<<"\n"<<H<<std::endl;
std::cout<<"F:"<<m_F.size<<"\n"<<m_F<<std::endl;
std::cout<<"H*F:"<<realOutput.size<<"\n"<<H*m_F<<std::endl;
*/
}

void ELM_IN_ELM_Model::init_greedyFitWhole(int g)
{
    int n_models = m_n_models;
    m_subModels.clear();
    
    ELM_Model subModel;
    subModel.inputData_2d(m_trainImgs,m_trainLabelBins,m_width,m_height,m_channels);
    subModel.inputData_2d_test(m_testImgs,m_testLabelBins);
    
    float maxScore = 0;
    for(int n=1;n<=n_models;n++)
    {
        m_n_models = n;
        m_subModels.resize(n);
        
        subModel.setHiddenNodes(m_subModelHiddenNodes[n-1]);
        
        ELM_Model maxSubModel;

        for(int i=0;i<g;i++)
        {
            subModel.fit(-1,false);
            m_subModels[n-1] = subModel;
            fitMainModel(-1,false);
            
            float score = validate();
            if(score > maxScore)
            {
                std::cout<<"g score:"<<score<<std::endl;
                maxScore = score;
                maxSubModel = subModel;
            }
            
            subModel.clear();
            m_K.release();
            m_F.release();
        }
        
        m_subModels[n-1] = maxSubModel;
    }
    
    std::cout<<"final score:"<<maxScore<<std::endl;
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
    cv::FileStorage fswrite(m_modelPath+"mainModel.xml",cv::FileStorage::WRITE);
    
    fswrite<<"n_models"<<m_n_models;
    fswrite<<"subModelPath"<<m_modelPath;
    fswrite<<"width"<<m_width;
    fswrite<<"height"<<m_height;
    fswrite<<"channels"<<m_channels;
    fswrite<<"C"<<m_C;
    fswrite<<"F"<<m_F;
    fswrite<<"label_string"<<m_label_string;
    
    fswrite.release();
    
    cv::FileStorage K_fswrite(m_modelPath+"mainK.xml",cv::FileStorage::WRITE);
    K_fswrite<<"K"<<m_K;
    K_fswrite.release();
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

    fsread.release();
    
    cv::FileStorage K_fsread(m_modelPath+"mainK.xml",cv::FileStorage::READ);
    K_fsread["K"]>>m_K;
    K_fsread.release();

    //加载子模型
    m_subModels.resize(m_n_models);
    for(int m=0;m<m_n_models;m++)
        m_subModels[m].load(m_modelPath+"subModel"+std::to_string(m)+".xml",
                            m_modelPath+"subK"+std::to_string(m)+".xml");
}

void ELM_IN_ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
        normalize(ROI);
    }
    
    cv::Mat output = H * m_F;
    //std::cout<<output<<std::endl;
    int maxId = getMaxId(output);
    
    label.assign(m_label_string[maxId]);
}

void ELM_IN_ELM_Model::clearTrainData()
{
    m_subModelToTrain.clearTrainData();
    for(int i=0;i<m_n_models;i++)
        m_subModels[i].clearTrainData();
    
    m_trainImgs.clear();
    m_trainLabelBins.clear();
    m_testImgs.clear();
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
    
    fitSubModels();
    fitMainModel();
}
