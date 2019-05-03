#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    
    std::string strArgv(argv[1]);
    if(strArgv.find("updatedb") == std::string::npos)
    {
        std::string videoName(argv[2]);
        w.setVideo(videoName);
    }

    if(strArgv == "resnet" || strArgv == "elm")
        w.setMethod(strArgv);
    else if(strArgv == "updatedb")
    {
        handleFaceDb(1);
        handleFaceDb(2);
        return 0;
    }
    else if(strArgv == "updatedb-elm")
    {
        handleFaceDb(1);
        return 0;
    }
    else if(strArgv == "updatedb-resnet")
    {
        handleFaceDb(2);
        return 0;
    }
    else
    {
        std::cout<<"parameters error!"<<std::endl;
        return 0;
    }
    
    w.show();
    
    return a.exec();
}
