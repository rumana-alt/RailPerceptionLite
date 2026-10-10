#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

// ---- Tunable settings ----
const double HALF_W_BOTTOM = 0.12;  // corridor half-width at bottom
const double HALF_W_TOP    = 0.02;  // corridor half-width at top
const double X_SHIFT       = -0.06;  // move corridor left
const float  SMOOTH        = 0.2f;  // smaller = smoother

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

// Fit one line x = a*y + b through all segment endpoints
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

// Is a point inside the hazard zone? (for the YOLO step later)
bool inHazardZone(const std::vector<cv::Point>& zone, cv::Point pt) {
    if (zone.size() < 3) return false;
    bool inside = false;
    for (size_t i = 0, j = zone.size() - 1; i < zone.size(); j = i++) {
        double xi = zone[i].x, yi = zone[i].y;
        double xj = zone[j].x, yj = zone[j].y;
        bool crosses = ((yi > pt.y) != (yj > pt.y)) &&
                       (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
        if (crosses) inside = !inside;
    }
    return inside;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../data/test.mp4";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << path << std::endl;
        return 1;
    }

    cv::Mat frame, small, gray, blur, edges, roi, view, overlay;
    int n = 0;
    bool init = false;
    float sxb = 0, sxt = 0;

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
        int W = view.cols, H = view.rows;
        cv::polylines(view, roiPolygon(W, H), true, cv::Scalar(0, 255, 0), 1);

        std::vector<cv::Vec4i> steep;
        std::vector<double> midX;
        for (const auto& l : lines) {
            double dx = l[2] - l[0], dy = l[3] - l[1];
            double angle = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
            if (angle < 40 || angle > 140) continue;
            steep.push_back(l);
            midX.push_back((l[0] + l[2]) / 2.0);
        }

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

        int yTop = (int)(H * 0.58), yBot = H - 1;
        cv::Point a1, a2, b1, b2;
        bool okL = fitRailLine(left, yTop, yBot, a1, a2);
        bool okR = fitRailLine(right, yTop, yBot, b1, b2);
        if (okL) cv::line(view, a1, a2, cv::Scalar(0, 0, 255), 2);
        if (okR) cv::line(view, b1, b2, cv::Scalar(255, 0, 0), 2);

        float xb = 0, xt = 0;
        bool have = false;
        if (okL && okR) { xb = (a1.x + b1.x) / 2.0f; xt = (a2.x + b2.x) / 2.0f; have = true; }
        else if (okL)   { xb = (float)a1.x; xt = (float)a2.x; have = true; }
        else if (okR)   { xb = (float)b1.x; xt = (float)b2.x; have = true; }

        if (have) {
            if (!init) { sxb = xb; sxt = xt; init = true; }
            else {
                sxb = (1 - SMOOTH) * sxb + SMOOTH * xb;
                sxt = (1 - SMOOTH) * sxt + SMOOTH * xt;
            }
        }

        std::vector<cv::Point> zone;
        if (init) {
            float shift = (float)(X_SHIFT * W);
            float wb = (float)(HALF_W_BOTTOM * W);
            float wt = (float)(HALF_W_TOP * W);
            zone = {
                cv::Point((int)(sxb + shift - wb), yBot),
                cv::Point((int)(sxt + shift - wt), yTop),
                cv::Point((int)(sxt + shift + wt), yTop),
                cv::Point((int)(sxb + shift + wb), yBot)
            };
            overlay = view.clone();
            cv::fillConvexPoly(overlay, zone, cv::Scalar(0, 255, 0));
            cv::addWeighted(overlay, 0.35, view, 0.65, 0, view);
            cv::polylines(view, zone, true, cv::Scalar(0, 255, 255), 2);
        }

        cv::putText(view, init ? "Hazard zone: ON" : "Hazard zone: searching...",
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        cv::imshow("1 Hazard Zone", view);
        cv::imshow("3 ROI Edges", roi);

        if (n % 50 == 0 && n <= 200) {
            cv::imwrite("../hazard_" + std::to_string(n) + ".png", view);
            std::cout << "Frame " << n << ": L " << left.size()
                      << ", R " << right.size()
                      << ", zone " << (init ? "ok" : "none") << std::endl;
        }

        int key = cv::waitKey(30);
        if (key == 'p') key = cv::waitKey(0);
        if (key == 27 || key == 'q') break;
        n++;
    }
    return 0;
}