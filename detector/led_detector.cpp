#include "led_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

// 比较函数，接受重心作为参数
bool comparePointsWithCenter(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& center) {
    double angle_a=std::atan2(a.y -center.y,a.x-center.x);
    double angle_b=std::atan2(b.y -center.y,b.x-center.x);
    return angle_a<angle_b;  //计算正切，来比较a与b相对于中心的角度，这样如果b的角度更大让a在b前面，从而逆时针排序
}

static void orderCorners(cv::Point2f pts[4])//pts可以说是指向cv::Point2f的一个指针
 {
    //计算重心
    cv::Point2f center = (pts[0]+pts[1] +pts[2]+pts[3])/4.0f;
    //事实上是把x，y分别做了平均数找到了中心点
    
    //排序
    for (int i=0;i<4;i++) {
        for (int j =i+1;j<4;j++) {
            if (!comparePointsWithCenter(pts[i], pts[j], center)) {
                std::swap(pts[i], pts[j]);  
            }
        }
    }
}
//这个函数是我们用来防止后面，连线的时候无序
  

LEDdetector::LEDdetector(){
    std::cout<<"初始化完成 ——"<<std::endl;
}

//预处理，把红色通道图分离出来二值化
cv::Mat LEDdetector::channelBinary(const cv::Mat& src)
{
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red1, red2;//由于红色有两个区间我们需要将0~10和160~180都分离出来再合并
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), red2);//H 色相  S 饱和度  V 亮度

    return red1 | red2;
}

//形态学开运算，先腐蚀后膨胀
cv::Mat LEDdetector::morphOpen(const cv::Mat& bin)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,  cv::Size(MORPH_KSIZE, MORPH_KSIZE));//必须配合着getStructuringElement的函数先生成形态学核
    cv::Mat result;
    cv::morphologyEx(bin, result, cv::MORPH_OPEN, kernel);
    return result;
}

//筛选灯条，卡一下条件
std::vector<cv::RotatedRect> LEDdetector::findLights(const cv::Mat& bin)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> lights;
    for (int i = 0; i < contours.size(); i++){
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        double area = rect.size.width * rect.size.height;
        if (area < LIGHT_MIN_AREA) continue;
        //最小面积过滤一下

        double long_side  = std::max(rect.size.height, rect.size.width);
        double short_side = std::min(rect.size.height, rect.size.width);
        double ratio      = long_side/short_side;
        //长宽比定义，长的除以宽的

        //最大最小长宽比过滤一下
        if (ratio>LIGHT_MAX_RATIO || ratio<LIGHT_MIN_RATIO) continue;

        //角度过滤一下
        if (std::abs(rect.angle)>LIGHT_MAX_ANGLE) //abs绝对值，因为你ninAreaRect的返回角度是负值
        continue;
       
        lights.push_back(rect);//筛选出来了符合全部条件的灯条轮廓并且返回
    }
    return lights;
}

//机甲板配对
std::vector<std::vector<cv::Point2f>>
LEDdetector::matchArmor(const std::vector<cv::RotatedRect>& lights)
{
    std::vector<std::vector<cv::Point2f>> all_corners;

    for (int i=0;i<lights.size();i++){
        for (int j = i + 1; j < lights.size(); j++){
            const cv::RotatedRect& left  = lights[i];
            const cv::RotatedRect& right = lights[j];

            //角度差过滤
            double angleDiff = std::abs(left.angle - right.angle);
            if (angleDiff > MATCH_ANGLE_DIFF) continue;

            //长度差过滤
            double hl = std::max(left.size.height, left.size.width);
            double hr = std::max(right.size.height, right.size.width);
            double height_ratio = (hl > hr) ? hl / hr : hr / hl;
            if (height_ratio > MATCH_HEIGHT_RATIO) continue;

            //距离差过滤
            double dist = cv::norm(left.center - right.center);
            double maxDist = (hl + hr) * 0.5 * MATCH_CENTER_DIST_RATIO;
            if (dist > maxDist) continue;

            //对于四点开始排序
            cv::Point2f left_pts[4], right_pts[4];
            left.points(left_pts);right.points(right_pts);
            orderCorners(left_pts);
            orderCorners(right_pts);

            //从而从两个机甲板灯条八个顶点筛选出来四个顶点
            std::vector<cv::Point2f> corners = {
                left_pts[0],   // 左上
                right_pts[1],  // 右上
                right_pts[2],  // 右下
                left_pts[3]    // 左下
            };
            all_corners.push_back(corners);
        }
    }
    return all_corners;
}

//主入口，并且最终可视化画出图像
void LEDdetector::detectAndDraw(cv::Mat& image)
{
    double start_time=cv::getTickCount();

    cv::Mat binary=channelBinary(image);
    binary=morphOpen(binary);
    cv::imshow("Binary",binary); // 调试窗口

    std::vector<cv::RotatedRect> lights = findLights(binary);
    std::vector<std::vector<cv::Point2f>> all_corners = matchArmor(lights);

    double end_time = cv::getTickCount();
    double process_time = (end_time - start_time) * 1000 / cv::getTickFrequency();
    std::cout << "灯条: " << lights.size()<< ", 装甲板: " << all_corners.size() << ", 处理时间: " << process_time << " ms" << std::endl;
    

//画出绿色矩形
for (int i = 0; i < all_corners.size(); i++){
    std::vector<cv::Point2f> corners = all_corners[i];
    for (int j = 0; j < 4; j++)
        cv::line(image, corners[j], corners[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
}
}