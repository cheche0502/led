cmake_minimum_required(VERSION 3.10)
project(RedVertex)

find_package(OpenCV REQUIRED)

add_executable(tju
    detector/main.cpp 
    detector/led_detector.cpp
)

target_link_libraries(tju ${OpenCV_LIBS})