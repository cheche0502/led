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
    // 添加相机内参和畸变系数
    camera_matrix = (cv::Mat_<double>(3, 3) <<9.28130989e+02, 0, 3.77572945e+02, 0, 9.30138391e+02, 2.83892859e+02, 0, 0, 1.0);
    dist_coeffs = (cv::Mat_<double>(5, 1) << -2.54433647e-01, 5.69431382e-01, 3.65405229e-03,-1.09433818e-03, -1.33846840e+00);

    // 添加装甲板3D坐标
    double armor_width = 0.130;
    double armor_height = 0.050;
    
    armor_3d_points =    //装甲板中心点是（0，0）
     {
        cv::Point3f(-armor_width/2, -armor_height/2, 0),
        cv::Point3f(armor_width/2, -armor_height/2, 0),
        cv::Point3f(armor_width/2, armor_height/2, 0),
        cv::Point3f(-armor_width/2, armor_height/2, 0)
    };
    std::cout<<"初始化完成 ——"<<std::endl;
}

//预处理，把红色通道图分离出来二值化
cv::Mat LEDdetector::channelBinary(const cv::Mat& src)
{
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);//BGR转HSV颜色空间

    cv::Mat red1, red2;//由于红色有两个区间我们需要将0~10和160~180都分离出来再合并
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), red2);//H 色相  S 饱和度  V 亮度

    return red1 | red2;
}

//形态学开运算，先腐蚀后膨胀
cv::Mat LEDdetector::morphOpen(const cv::Mat& bin)
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(MORPH_KSIZE, MORPH_KSIZE));//必须配合着getStructuringElement的函数先生成形态学核
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
        if (std::abs(rect.angle)>LIGHT_MAX_ANGLE) //abs绝对值，因为你minAreaRect的返回角度是负值
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
        for (int j = i + 1;j<lights.size();j++){
            const cv::RotatedRect& left = lights[i];
            const cv::RotatedRect& right = lights[j];

            //角度差过滤
            double angleDiff = std::abs(left.angle-right.angle);
            if (angleDiff > MATCH_ANGLE_DIFF) continue;

            //高度比过滤
            double hl = std::max(left.size.height, left.size.width);
            double hr = std::max(right.size.height, right.size.width);
            double height_ratio = max(hl/hr,hr/hl);
            if (height_ratio > MATCH_HEIGHT_RATIO) continue;

            //距离差过滤
            double dist = cv::norm(left.center - right.center);
            double maxDist = (hl+hr)*0.5*MATCH_CENTER_DIST_RATIO;
            if (dist>maxDist) continue;

            //对于四点开始排序
            cv::Point2f left_pts[4], right_pts[4];
            left.points(left_pts);right.points(right_pts);
            orderCorners(left_pts);
            orderCorners(right_pts);

            //从而从两个机甲板灯条八个顶点筛选出来四个顶点
            std::vector<cv::Point2f> corners = 
            {
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
void LEDdetector::detectAndDraw(cv::Mat& image) {
    double start_time = cv::getTickCount();

    // 图像预处理
    cv::Mat binary = channelBinary(image);
    binary = morphOpen(binary);
    cv::imshow("Binary", binary);

    // 检测灯条和装甲板
    std::vector<cv::RotatedRect> lights = findLights(binary);
    std::vector<std::vector<cv::Point2f>> all_corners = matchArmor(lights);

    // 对每个检测到的装甲板进行处理
    for (int i = 0;i<all_corners.size();i++) {
        std::vector<cv::Point2f> corners = all_corners[i];
        
        // 检查角点是否有效（在图像范围内）
        bool corners_valid = true;
        for (int j = 0; j < 4; j++) {
            if (corners[j].x < 0 || corners[j].x >= image.cols || corners[j].y < 0 || corners[j].y >= image.rows)
            {
                corners_valid = false; break;
            }
        }
        
        if (corners_valid) {
            // 绘制装甲板框
            for (int j = 0; j < 4; j++) {
                cv::line(image,corners[j],corners[(j+1)%4],cv::Scalar(0,255,0),2);
            }
            
            // 进行PnP解算
            cv::Vec3d rvec, tvec;
            if (solveArmorPose(corners, rvec, tvec)) {
                // 显示距离信息
                double distance = cv::norm(tvec);
                std::string dist_text ="Dist: "+std::to_string(distance).substr(0,4) +"m";
                cv::putText(image, dist_text, cv::Point(20, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                // 绘制3D坐标轴
                float axis_length =0.15f; // 15厘米坐标轴
                std::vector<cv::Point3f> axis_3d = {
                    cv::Point3f(0, 0, 0),
                    cv::Point3f(axis_length, 0, 0),
                    cv::Point3f(0, axis_length, 0),
                    cv::Point3f(0, 0, axis_length)
                };
                
                std::vector<cv::Point2f> axis_2d;//存放投影后的二维点
                cv::projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs, axis_2d);
                //参数说明：3D点，旋转向量，平移向量，相机内参，畸变系数，输出的2D点
                
                // 绘制坐标轴
                cv::arrowedLine(image, axis_2d[0], axis_2d[1], cv::Scalar(0, 0, 255), 3); // X-红
                cv::arrowedLine(image, axis_2d[0], axis_2d[2], cv::Scalar(0, 255, 0), 3); // Y-绿
                cv::arrowedLine(image, axis_2d[0], axis_2d[3], cv::Scalar(255, 0, 0), 3); // Z-蓝
            }
        }
    }
}
// PnP位姿解算函数 - 放在主函数后面
bool LEDdetector::solveArmorPose(const std::vector<cv::Point2f>& corners_2d, cv::Vec3d& rvec, cv::Vec3d& tvec) 
{
    // 检查是否有4个角点
    if (corners_2d.size() != 4) {
        return false;
    }
    
    // 复制角点并排序
    std::vector<cv::Point2f> ordered_corners = corners_2d;
    orderCorners(ordered_corners.data());
    
    // 使用EPnP方法计算位姿
    bool success = cv::solvePnP(armor_3d_points, ordered_corners, camera_matrix, dist_coeffs,rvec, tvec, false, cv::SOLVEPNP_EPNP);
    //参数说明：3D点，2D点，相机内参，畸变系数，输出的旋转向量，输出的平移向量，是否使用初始估计，使用的方法
    
    // 如果成功，显示结果
    if (success) {
        double distance = cv::norm(tvec);
        std::cout << "=== 装甲板位姿 ===" << std::endl;
        std::cout << "位置(X,Y,Z): " << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << " m" << std::endl;
        std::cout << "到相机距离: " << distance << " m" << std::endl;
    }
    
    return success;
}
