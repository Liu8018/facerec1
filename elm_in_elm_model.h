#ifndef ELM_IN_ELM_MODEL_H
#define ELM_IN_ELM_MODEL_H

#include "elm_model.h"

class ELM_IN_ELM_Model
{
public:
    ELM_IN_ELM_Model();
    ELM_IN_ELM_Model(const int n_models,const std::string modelDir);
    
    void setInitPara(const int n_models,const std::string modelDir);
    
    void setSubModelHiddenNodes(const int modelId, const int n_nodes);
    
    void loadStandardDataset(const std::string path, const float trainSampleRatio,
                             const int resizeWidth, const int resizeHeight, 
                             const int channels, bool shuffle=true);
    
    void loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle=true);
    
    void fitSubModels(int batchSize = -1, bool validating = true);
    void fitMainModel(int batchSize = -1, bool validating = true);
    
    void save();
    void load(std::string modelDir);
    
    void query(const cv::Mat &mat, std::string &label);
    
    //计算在测试数据上的准确率
    float validate();
    
    void init_greedyFitWhole(int g);
    
private:
    int m_n_models;
    std::vector<int> m_subModelHiddenNodes;
    
    ELM_Model m_subModelToTrain;
    
    std::vector<ELM_Model> m_subModels;
    
    std::string m_modelPath;
    int m_width;
    int m_height;
    int m_channels;
    int m_Q;
    int m_C;
    
    cv::Mat m_K;
    
    cv::Mat m_F;
    
    std::vector<std::string> m_label_string;
    
    std::vector<cv::Mat> m_trainImgs;
    std::vector<cv::Mat> m_testImgs;
    std::vector<std::vector<bool>> m_trainLabelBins;
    std::vector<std::vector<bool>> m_testLabelBins;
    
};

#endif // ELM_IN_ELM_MODEL_H
