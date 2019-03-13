#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SignUpDialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    //初始化：摄像头
    m_capture.open("/dev/video0");
    
    if (!m_capture.isOpened())
    {
        std::cout << "Can't capture!" << std::endl;
        exit(0);
    }
    
    //初始化：检测器
    m_detection.Load("./data/cascade/haar_face_0.xml");
    SimdDetection::Size frameSize(m_capture.get(3),m_capture.get(4));
    m_detection.Init(frameSize, 1.2, frameSize / 5);
    
    //初始化：人脸识别
    m_rec = FaceRecognition("resnet");
    
    if(m_rec.method == "resnet")
    {
        //resnet人脸识别初始化
        if(access("./data/face_database/faceDescriptors.dat",F_OK) == -1)
            m_rec.init_updatedb();
        else
            m_rec.init_loadDb();
    }
    
    if(m_rec.method == "elm")
    {
        //ELM_IN_ELM人脸识别初始化
        if(access("./data/ELM_Models/mainModel.xml",F_OK) == -1)
            m_rec.init_updateEIEdb();
        else
            m_rec.init_loadEIEdb();
    }
    
    m_isDoFaceRec = false;
    m_faceRecKeepTime = 5;
    
    //将timer与getframe连接
    connect(m_timer,SIGNAL(timeout()),this,SLOT(updateFrame()));
    m_timer->setInterval(1000/m_capture.get(cv::CAP_PROP_FPS));
    m_timer->start();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_timer;
}

void MainWindow::updateFrame()
{
    m_capture >> m_frameSrc;
    cv::flip(m_frameSrc,m_frameSrc,1);
    m_frameSrc.copyTo(m_frame);
    
    //类型转化
    SimdDetection::View image = m_frame;
    
    //进行检测
    SimdDetection::Objects objects;
    m_detection.Detect(image, objects);
    
    if(!objects.empty())
    {
        m_faceROI = m_frameSrc(objects[0].rect);
        
        //绘制检测结果
        Simd::DrawRectangle(image, objects[0].rect, Simd::Pixel::Bgr24(0, 255, 255),2);
        
        if(m_isDoFaceRec)
        {
            std::string name;
            bool isInFaceDb = false;
            
            if(m_rec.method == "resnet")
            {
                //人脸对齐
                dlib::full_object_detection shape;
                m_alignment.getShape(m_frameSrc,objects[0].rect,shape);
                
                //人脸识别
                isInFaceDb = m_rec.recognize(m_frameSrc,shape,name);
            }
            
            if(m_rec.method == "elm")
                isInFaceDb = m_rec.recognize(m_faceROI,name);
            
            //显示识别结果
            if(isInFaceDb)
                cv::putText(m_frame,name,objects[0].rect.TopLeft(),1,2,cv::Scalar(255,100,0),2);
            else
                cv::putText(m_frame,"others",objects[0].rect.TopLeft(),1,2,cv::Scalar(255,100,0),2);
            
            //到达持续时间，停止检测
            float keptTime = (cv::getTickCount()-m_faceRecStartTick)/cv::getTickFrequency();
            if(keptTime > m_faceRecKeepTime)
                m_isDoFaceRec = false;
        }
    }
    
    showMat();
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

void MainWindow::on_pushButton_Recognize_clicked()
{
    m_isDoFaceRec = true;
    m_faceRecStartTick = cv::getTickCount();
}

void MainWindow::on_pushButton_SignUp_clicked()
{
    if(m_faceROI.empty())
        return;
    
    m_timer->stop();
    
    //创建登记窗口
    SignUpDialog *signUpDlg = new SignUpDialog();
    //登记窗口打开时停止其他窗口的运行
    signUpDlg->setWindowModality(Qt::ApplicationModal);
    //信息传递
    connect(signUpDlg, SIGNAL(sendData(bool, std::string)), this, SLOT(addFace(bool, std::string)));
    
    signUpDlg->setImg(m_faceROI);
    signUpDlg->show();
    signUpDlg->exec();
    
    delete signUpDlg;
    
    m_timer->start();
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
        filename += name + ".jpg";
    }
    else
    {
        time_t t = time(0);
        char strTime[64];
        strftime(strTime, 64, "%Y-%m-%d-%H-%M-%S", localtime(&t));
        
        filename += name+std::string(strTime) + ".jpg";
    }
    
    cv::imwrite(filename,m_faceROI);
    
    //更新数据库
    if(m_rec.method == "resnet" && isNewClass)
    {
        m_rec.init_updatedb();
    }
    if(m_rec.method == "elm")
    {
        if(isNewClass)
            m_rec.init_updateEIEdb();
        else
            m_rec.updateEIEdb(m_faceROI,name);
    }
}
