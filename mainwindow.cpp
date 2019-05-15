#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SignUpDialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    ui->label_names->setStyleSheet("background:transparent;color:blue");
    ui->label_names->setFont(QFont("Microsoft YaHei", 18, 75));
    ui->label_names->setAlignment(Qt::AlignTop);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_timer;
}

void MainWindow::showMat()
{
    cv::Mat frameToShow;
    cv::resize(m_frame,frameToShow,cv::Size(ui->label_Main->width()-2,ui->label_Main->height()-2));

    cv::cvtColor(frameToShow,frameToShow,cv::COLOR_BGR2RGB);
    m_qimgFrame = QImage((const uchar*)frameToShow.data,
                  frameToShow.cols,frameToShow.rows,
                  frameToShow.cols*frameToShow.channels(),
                  QImage::Format_RGB888);

    ui->label_Main->setPixmap(QPixmap::fromImage(m_qimgFrame));
}

void MainWindow::showNames(std::map<float,std::string> nameScores)
{
    QString qstr;
    //反向遍历并输出
    for(std::map<float,std::string>::reverse_iterator it = nameScores.rbegin();it!=nameScores.rend();it++)
    {
        qstr.append(it->second.data());
        qstr.append(":");
        qstr.append(std::to_string(it->first).substr(0,6).data());
        qstr.append("\n");
    }
    ui->label_names->setText(qstr);
}

void MainWindow::on_pushButton_Recognize_clicked()
{
    m_isDoFaceRec = true;
    m_faceRecStartTick = cv::getTickCount();
}

void MainWindow::on_pushButton_SignUp_clicked()
{
    if(m_faceROI_src.empty())
        return;
    
    m_timer->stop();
    
    //创建登记窗口
    SignUpDialog *signUpDlg = new SignUpDialog();
    //登记窗口打开时停止其他窗口的运行
    signUpDlg->setWindowModality(Qt::ApplicationModal);
    //信息传递
    connect(signUpDlg, SIGNAL(sendData(bool, std::string)), this, SLOT(addFace(bool, std::string)));
    
    signUpDlg->setImg(m_faceROI_src);
    signUpDlg->show();
    signUpDlg->exec();
    
    delete signUpDlg;
    
    m_timer->start();
}

void MainWindow::setVideo(std::string video)
{
    m_video = video;
    
    //初始化：摄像头
    m_capture.open(m_video);
    
    if (!m_capture.isOpened())
    {
        std::cout << "Can't capture "<<m_video<< std::endl;
        exit(0);
    }
    
    m_isDoFaceRec = false;
    m_faceRecKeepTime = 500;
    
    //将timer与getframe连接
    connect(m_timer,SIGNAL(timeout()),this,SLOT(updateFrame()));
    m_timer->setInterval(1000/m_capture.get(cv::CAP_PROP_FPS));
    m_timer->start();
}

void MainWindow::setMethod(std::string method)
{
    //初始化：人脸识别
    //m_rec = FaceRecognition("resnet");
    //m_rec = FaceRecognition("elm");
    m_rec = FaceRecognition(method);
    
    if(m_rec.method == "resnet")
    {
        //resnet人脸识别初始化
        if(access("./data/face_database/faceDescriptors.dat",F_OK) == -1)
            m_rec.init_updateResnetDb();
        else
            m_rec.init_loadResnetDb();
    }
    
    if(m_rec.method == "elm")
    {
        //ELM_IN_ELM人脸识别初始化
        if(access("./data/ELM_Models/mainModel.xml",F_OK) == -1)
            m_rec.init_updateEIEdb();
        else
            m_rec.init_loadEIEdb();
        
        std::map<std::string, std::string> files;
        getFiles("./data/face_database",files);
        if(files.empty())
            isEmptyRun = true;
        else
            isEmptyRun = false;
    }
}

void MainWindow::updateFrame()
{
    m_capture >> m_frameSrc;
    cv::flip(m_frameSrc,m_frameSrc,1);
    m_frameSrc.copyTo(m_frame);
    
    //进行检测
    std::vector<cv::Rect> objects;
    m_detection.detect(m_frame, objects);
    
    if(!objects.empty())
    {
        //人脸对齐
        m_alignment.alignFace(m_frameSrc,objects[0],m_faceROI);
        m_faceRect = objects[0];
        m_faceROI_src = m_frameSrc(m_faceRect);
        
        cv::cvtColor(m_faceROI,m_faceROI,cv::COLOR_BGR2GRAY);
        //cv::imshow("m_faceROI",m_faceROI);
        cv::equalizeHist(m_faceROI,m_faceROI);
        cv::imshow("m_faceROI_eq",m_faceROI);
        
        //绘制检测结果
        cv::rectangle(m_frame,objects[0],cv::Scalar(0,255,255),2);
        
        if(m_isDoFaceRec)
        {
            std::string name;
            bool isInFaceDb = false;
            
            if(m_rec.method == "resnet")
            {
                dlib::full_object_detection shape;
                m_alignment.getShape(m_frameSrc,objects[0],shape);
                isInFaceDb = m_rec.recognize(m_frameSrc,shape,name);
            }
            
            if(m_rec.method == "elm")
            {
                int n = 5;
                std::map<float,std::string> nameScores;
                isInFaceDb = m_rec.recognize(m_faceROI,n,nameScores);
                
                showNames(nameScores);
                
                //for(int i=0;i<n;i++)
                //    std::cout<<"names["<<i<<"]:"<<names[i]<<std::endl;
                
                if(!nameScores.empty())
                    name = nameScores.rbegin()->second;
                else
                    name = "others";
            }
            
            //显示识别结果
            if(isInFaceDb)
                cv::putText(m_frame,name,objects[0].tl(),1,2,cv::Scalar(255,100,0),2);
            else
                cv::putText(m_frame,"others",objects[0].tl(),1,2,cv::Scalar(255,100,0),2);
            
            //到达持续时间，停止检测
            float keptTime = (cv::getTickCount()-m_faceRecStartTick)/cv::getTickFrequency();
            if(keptTime > m_faceRecKeepTime)
                m_isDoFaceRec = false;
        }
    }
    else
    {
        //ui->label_names->clear();
    }
    
    showMat();
}

void MainWindow::addFace(bool isSignUp, std::string name)
{
    if(!isSignUp || name.empty())
        return;
        
    std::string filename = "./data/face_database/" + name;
    
    //若不存在则新建
    bool isNewClass = 0;
    if(access(filename.data(),F_OK) == -1)
    {
        mkdir(filename.data(),00777);
        isNewClass = 1;
    }
    
    filename += "/";
    
    //输出
    if(isNewClass)
    {
        filename += name + ".png";
    }
    else
    {
        time_t t = time(0);
        char strTime[64];
        strftime(strTime, 64, "%Y-%m-%d-%H-%M-%S", localtime(&t));
        
        filename += name+std::string(strTime) + ".png";
    }
    
    markImg(m_faceROI_src);
    cv::imwrite(filename,m_faceROI_src);
    if(isEmptyRun)
    {
        filename = "./data/face_database/" + name + "/" + name + "2.png";
        cv::imwrite(filename,m_faceROI_src);
        isEmptyRun = false;
    }
    
    //更新数据库
    if(m_rec.method == "resnet" && isNewClass)
    {
        m_rec.init_updateResnetDb();
    }
    if(m_rec.method == "elm")
    {
        if(isNewClass)
            m_rec.init_updateEIEdb();
        else
            m_rec.updateEIEdb(m_faceROI,name);
    }
}
