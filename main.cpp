#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    
    if(argc == 1)
    {
        std::cout<<"parameters error!"<<std::endl;
        return 0;
    }
    
    std::string strArgv(argv[1]);
    if(strArgv == "resnet" || strArgv == "elm")
        w.setMethod(strArgv);
    else if(strArgv == "updatedb")
    {
        handleFaceDb();
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
