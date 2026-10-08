#include <opencv2/opencv.hpp>
#include <iostream>

// Keep only the bottom trapezoid of the frame (where rails usually are)
cv::Mat applyROI(const cv::Mat& edges) {
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    int w = edges.cols, h = edges.rows;
    std::vector<cv::Point> poly = {
        cv::Point((int)(w * 0.05), h),
        cv::Point((int)(w * 0.40), (int)(h * 0.55)),
        cv::Point((int)(w * 0.60), (int)(h * 0.55)),
        cv::Point((int)(w * 0.95), h)
    };
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    cv::Mat out;
    cv::bitwise_and(edges, mask, out);
    return out;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../data/test.mp4";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << path << std::endl;
        return 1;
    }

    cv::Mat frame, small, gray, blur, edges, roi;
    while (cap.read(frame)) {
        // Resize for speed
        double scale = 960.0 / frame.cols;
        cv::resize(frame, small, cv::Size(), scale, scale);

        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        cv::Canny(blur, edges, 50, 150);
        roi = applyROI(edges);

        cv::imshow("1 Original", small);
        cv::imshow("2 Edges", edges);
        cv::imshow("3 ROI Edges", roi);

        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) break;
    }
    return 0;
}