#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include "FaceDetection.h"
#include "functions.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
    void setMethod(std::string method);
    void setVideo(std::string video);
    
private:
    Ui::MainWindow *ui;
    
    std::string m_video;
    cv::VideoCapture m_capture;
    cv::Mat m_frame;
    cv::Mat m_frameSrc;
    
    FaceDetection m_detection;
    
    FaceAlignment m_alignment;
    
    FaceRecognition m_rec;
    cv::Mat m_faceROI;
    
    cv::Rect m_faceRect;
    
    QImage m_qimgFrame;
    QTimer *m_timer = new QTimer(this);
    
    void showMat();
    void showNames(std::map<float,std::string> nameScores);
    
    bool m_isDoFaceRec;
    float m_faceRecStartTick;
    float m_faceRecKeepTime; 
    
    bool isEmptyRun;
    
private slots:
    void updateFrame();
    void on_pushButton_Recognize_clicked();
    void on_pushButton_SignUp_clicked();
    void addFace(bool isSignUp, std::string name);
};

#endif // MAINWINDOW_H
