#include "mainwindow.h"
#include "ui_mainwindow.h"

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
    m_detection.Init(frameSize, 1.2, frameSize / 10);
    
    //初始化：人脸识别
    m_rec.init_updatedb();
    m_isDoFaceRec = false;
    m_faceRecKeepTime = 5;
    
    //将timer与getframe连接
    connect(m_timer,SIGNAL(timeout()),this,SLOT(updateFrame()));
    m_timer->setInterval(1000/m_capture.get(CV_CAP_PROP_FPS));
    m_timer->start();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_timer;
}

void MainWindow::updateFrame()
{
    m_capture >> m_frame;
    
    //类型转化（共享数据）
    SimdDetection::View image = m_frame;
    
    //进行检测
    SimdDetection::Objects objects;
    m_detection.Detect(image, objects);
    
    if(!objects.empty())
    {
        //人脸对齐
        dlib::full_object_detection shape;
        m_alignment.getShape(m_frame,objects[0].rect,shape);
        
        if(m_isDoFaceRec)
        {
            //人脸识别
            std::string name;
            bool isInFaceDb = m_rec.recognize(m_frame,shape,name);
            
            //绘制检测结果
            Simd::DrawRectangle(image, objects[0].rect, Simd::Pixel::Bgr24(0, 255, 255),2);
            
            if(isInFaceDb)
            {
                //显示识别结果
                cv::putText(m_frame,name,objects[0].rect.TopLeft(),1,2,cv::Scalar(255,100,0),2);
            }
            
            float keptTime = (cv::getTickCount()-m_faceRecStartTick)/cv::getTickFrequency();
            if(keptTime > m_faceRecKeepTime)
                m_isDoFaceRec = false;
        }
    }
    
    showMat();
}

void MainWindow::showMat()
{
    cv::resize(m_frame,m_frame,cv::Size(ui->label_Main->width()-2,ui->label_Main->height()-2));

    cv::cvtColor(m_frame,m_frame,CV_BGR2RGB);
    m_qimgFrame = QImage((const uchar*)m_frame.data,
                  m_frame.cols,m_frame.rows,
                  m_frame.cols*m_frame.channels(),
                  QImage::Format_RGB888);

    ui->label_Main->setPixmap(QPixmap::fromImage(m_qimgFrame));
}

void MainWindow::on_pushButton_Recognize_clicked()
{
    m_isDoFaceRec = true;
    m_faceRecStartTick = cv::getTickCount();
}
