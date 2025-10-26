#include "led_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    //打开视频
    cv::VideoCapture cap("/home/cwy/led/armor.avi");
   
    if (!cap.isOpened())   {std::cout << "打不开视频\n";  return -1; }
   
   
   int fw = cap.get(cv::CAP_PROP_FRAME_WIDTH);
   int fh = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
   double fps = cap.get(cv::CAP_PROP_FPS);


    std::cout <<"帧宽×高 "<<fw<<"×"<< fh <<" FPS"<<fps<<std::endl;//确认

    //创建了一个输出的视频，并且他的基本信息应该与输入的装甲板视频相同，fps 长宽
    cv::VideoWriter rec("/home/cwy/led/armor_out.avi",cv::VideoWriter::fourcc('M','J','P','G'), fps,cv::Size(fw, fh));
    
    if (!rec.isOpened())   { std::cout << "无法录制\n"; return -1; }

    //实例化检测灯条的检测器
    LEDdetector det;

    //主循环
    cv::Mat frame;//frame来存放每一帧图像
    int idx = 0;
    while (cap.read(frame)){
        std::cout << "frame " << idx++ << '\r' << std::flush;
        if (frame.empty()) break;

        det.detectAndDraw(frame); //调用我写的算法，进行操作

        rec.write(frame);//把每一帧写进视频文件
        cv::imshow("out", frame);//实时展示
        if (cv::waitKey(100) == 27) break;  
    }

    std::cout << "\n  正常结束，输出已保存为 armor_out.avi" << std::endl;
    cv::destroyAllWindows();
    return 0;
}
