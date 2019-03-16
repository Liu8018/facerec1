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
    
private:
    Ui::MainWindow *ui;
    
    cv::VideoCapture m_capture;
    cv::Mat m_frame;
    cv::Mat m_frameSrc;
    
    SimdDetection m_detection;
    
    FaceAlignment m_alignment;
    
    FaceRecognition m_rec;
    cv::Mat m_faceROI;
    
    QImage m_qimgFrame;
    QTimer *m_timer = new QTimer(this);
    
    void showMat();
    
    bool m_isDoFaceRec;
    float m_faceRecStartTick;
    float m_faceRecKeepTime; 
    
private slots:
    void updateFrame();
    void on_pushButton_Recognize_clicked();
    void on_pushButton_SignUp_clicked();
    void addFace(bool isSignUp, std::string name);
};

#endif // MAINWINDOW_H
