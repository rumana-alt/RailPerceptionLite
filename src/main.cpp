#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

// ROI polygon (bottom-right area where rails are in this video)
std::vector<cv::Point> roiPolygon(int w, int h) {
    return {
        cv::Point((int)(w * 0.40), h),
        cv::Point((int)(w * 0.70), (int)(h * 0.55)),
        cv::Point((int)(w * 0.85), (int)(h * 0.55)),
        cv::Point((int)(w * 0.85), h)
    };
}

cv::Mat applyROI(const cv::Mat& edges) {
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    std::vector<cv::Point> poly = roiPolygon(edges.cols, edges.rows);
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    cv::Mat out;
    cv::bitwise_and(edges, mask, out);
    return out;
}

// Fit one line x = a*y + b through all segment endpoints (least squares)
bool fitRailLine(const std::vector<cv::Vec4i>& segs, int yTop, int yBottom,
                 cv::Point& p1, cv::Point& p2) {
    if (segs.size() < 2) return false;
    double n = 0, sumX = 0, sumY = 0, sumXY = 0, sumYY = 0;
    for (const auto& s : segs) {
        for (int k = 0; k < 2; k++) {
            double x = s[2 * k], y = s[2 * k + 1];
            n++; sumX += x; sumY += y; sumXY += x * y; sumYY += y * y;
        }
    }
    double denom = n * sumYY - sumY * sumY;
    if (std::abs(denom) < 1e-6) return false;
    double a = (n * sumXY - sumY * sumX) / denom;
    double b = (sumX - a * sumY) / n;
    p1 = cv::Point((int)(a * yBottom + b), yBottom);
    p2 = cv::Point((int)(a * yTop + b), yTop);
    return true;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../data/test.mp4";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << path << std::endl;
        return 1;
    }

    cv::Mat frame, small, gray, blur, edges, roi, view;
    int n = 0;
    while (cap.read(frame)) {
        double scale = 600.0 / frame.rows;
        cv::resize(frame, small, cv::Size(), scale, scale);

        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        cv::Canny(blur, edges, 50, 150);
        roi = applyROI(edges);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(roi, lines, 1, CV_PI / 180, 50, 50, 8);

        view = small.clone();
        auto poly = roiPolygon(view.cols, view.rows);
        cv::polylines(view, poly, true, cv::Scalar(0, 255, 0), 1);

        // Keep steep segments only (drop near-horizontal sleepers)
        std::vector<cv::Vec4i> steep;
        std::vector<double> midX;
        for (const auto& l : lines) {
            double dx = l[2] - l[0], dy = l[3] - l[1];
            double angle = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
            if (angle < 40 || angle > 140) continue;
            steep.push_back(l);
            midX.push_back((l[0] + l[2]) / 2.0);
        }

        // Split into left/right rail by median x
        std::vector<cv::Vec4i> left, right;
        if (steep.size() >= 4) {
            std::vector<double> sorted = midX;
            std::sort(sorted.begin(), sorted.end());
            double med = sorted[sorted.size() / 2];
            for (size_t i = 0; i < steep.size(); i++) {
                if (midX[i] < med) left.push_back(steep[i]);
                else right.push_back(steep[i]);
            }
        }

        int yTop = (int)(view.rows * 0.58), yBot = view.rows - 1;
        cv::Point a1, a2, b1, b2;
        bool okL = fitRailLine(left, yTop, yBot, a1, a2);
        bool okR = fitRailLine(right, yTop, yBot, b1, b2);
        if (okL) cv::line(view, a1, a2, cv::Scalar(0, 0, 255), 3);   // red = left rail
        if (okR) cv::line(view, b1, b2, cv::Scalar(255, 0, 0), 3);   // blue = right rail

        cv::imshow("1 Original + Lines", view);
        cv::imshow("3 ROI Edges", roi);

        if (n % 50 == 0 && n <= 200) {
            cv::imwrite("../hough_" + std::to_string(n) + ".png", view);
            std::cout << "Frame " << n << ": segments " << lines.size()
                      << ", steep " << steep.size()
                      << ", L " << left.size() << ", R " << right.size() << std::endl;
        }

        int key = cv::waitKey(30);
        if (key == 'p') key = cv::waitKey(0);   // p = pause, next key = resume
        if (key == 27 || key == 'q') break;     // Esc or q = quit
        n++;
    }
    return 0;
}