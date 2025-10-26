
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>  
#include <iostream>
#include <algorithm>

class LEDdetector {
public:
    LEDdetector();
    void detectAndDraw(cv::Mat& image);

private:
    cv::Mat channelBinary(const cv::Mat& src);
    cv::Mat morphOpen(const cv::Mat& bin);
    std::vector<cv::RotatedRect> findLights(const cv::Mat& bin);
    std::vector<std::vector<cv::Point2f>> matchArmor(const std::vector<cv::RotatedRect>& lights);
   
    bool solveArmorPose(const std::vector<cv::Point2f>& corners_2d, cv::Vec3d& rvec, cv::Vec3d& tvec);
    // 完整的参数列表
    int CHAN_R_TH = 200;  //红色通道二值化阈值，超过这个亮度才会被保留，高了可能会导致漏检，低了可能会错误

    int MORPH_KSIZE = 3;  //形态学参数，开运算的核的大小

    double LIGHT_MIN_AREA = 30;   //灯条被认定的最小面积，防止噪点
    double LIGHT_MAX_RATIO = 10.0;  //灯条最大的长宽比，防止正方形和圆形灯条出现，如果这个值太小可能会导致细长的灯条出现
    double LIGHT_MIN_RATIO = 5.0;   //灯条最小长宽比
    double LIGHT_MAX_ANGLE = 45.0; //灯条最大角度，过滤掉横着的灯条，minAreaRect的角度范围是-90~0
    
    double MATCH_ANGLE_DIFF = 10.0;  //作为一个装甲板，两个灯条的方向基本一致，利用角度检验
    double MATCH_HEIGHT_RATIO = 3.0;   //确保高度相近，是利用高灯条除以矮灯条
    double MATCH_CENTER_DIST_RATIO = 2;   //灯条距离的合理性
    double MATCH_MIN_DISTANCE = 15.0; //灯条间最小距离，防止重叠灯条被误认为装甲板
    
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Point3f> armor_3d_points;
    
};