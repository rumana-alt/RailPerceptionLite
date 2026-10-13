#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

// ---- Tunable settings ----
const double HALF_W_BOTTOM = 0.12;
const double HALF_W_TOP    = 0.02;
const double X_SHIFT       = -0.06;  // + right, - left
const float  SMOOTH        = 0.2f;
const float  CONF_THRESH   = 0.35f;  // YOLO confidence
const float  NMS_THRESH    = 0.45f;
const int    DETECT_EVERY  = 3;      // run YOLO every Nth frame (speed)

struct Det {
    cv::Rect box;
    int cls;
    float conf;
};

// COCO classes we care about on a track
bool isHazardClass(int c) {
    return c == 0 || c == 1 || c == 2 || c == 3 || (c >= 15 && c <= 19);
}

std::string className(int c) {
    switch (c) {
        case 0: return "person";
        case 1: return "bicycle";
        case 2: return "car";
        case 3: return "motorcycle";
        case 15: return "cat";
        case 16: return "dog";
        case 17: return "horse";
        case 18: return "sheep";
        case 19: return "cow";
        default: return "obj";
    }
}

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

// Is a point inside the hazard zone? (ray casting)
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

// Run YOLOv8 (ONNX, 640x640, output 1x84x8400) on one image
std::vector<Det> detect(cv::dnn::Net& net, const cv::Mat& img) {
    const int S = 640;
    float r = std::min((float)S / img.cols, (float)S / img.rows);
    int nw = (int)std::round(img.cols * r);
    int nh = (int)std::round(img.rows * r);
    int padX = (S - nw) / 2, padY = (S - nh) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, nw, nh)));

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, cv::Size(S, S),
                                          cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<Det> result;
    if (outs.empty()) return result;
    cv::Mat& o = outs[0];
    if (o.dims != 3) return result;
    int numCh = o.size[1];     // 84
    int numBoxes = o.size[2];  // 8400
    const float* d = (const float*)o.data;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
    for (int i = 0; i < numBoxes; i++) {
        float best = 0; int bestC = -1;
        for (int c = 0; c < numCh - 4; c++) {
            if (!isHazardClass(c)) continue;
            float s = d[(4 + c) * numBoxes + i];
            if (s > best) { best = s; bestC = c; }
        }
        if (bestC < 0 || best < CONF_THRESH) continue;
        float cx = d[0 * numBoxes + i], cy = d[1 * numBoxes + i];
        float w = d[2 * numBoxes + i], h = d[3 * numBoxes + i];
        int x = (int)((cx - w / 2 - padX) / r);
        int y = (int)((cy - h / 2 - padY) / r);
        int bw = (int)(w / r), bh = (int)(h / r);
        cv::Rect box = cv::Rect(x, y, bw, bh) & cv::Rect(0, 0, img.cols, img.rows);
        if (box.width < 2 || box.height < 2) continue;
        boxes.push_back(box);
        scores.push_back(best);
        classes.push_back(bestC);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, keep);
    for (int k : keep) result.push_back({boxes[k], classes[k], scores[k]});
    return result;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../data/test.mp4";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << path << std::endl;
        return 1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX("../models/yolov8n.onnx");
    if (net.empty()) {
        std::cerr << "Cannot load model ../models/yolov8n.onnx" << std::endl;
        return 1;
    }

    cv::Mat frame, small, gray, blur, edges, roi, view, overlay;
    int n = 0, savedAlerts = 0;
    bool init = false;
    float sxb = 0, sxt = 0;
    std::vector<Det> dets;

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

        // YOLO detection (every Nth frame, reuse in between)
        if (n % DETECT_EVERY == 0) dets = detect(net, small);

        int alerts = 0;
        for (const auto& d : dets) {
            // bottom-centre of the box = where the object touches the ground
            cv::Point foot(d.box.x + d.box.width / 2, d.box.y + d.box.height);
            bool danger = inHazardZone(zone, foot);
            cv::Scalar color = danger ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 200, 0);
            cv::rectangle(view, d.box, color, 2);
            std::string label = className(d.cls) + " " + std::to_string((int)(d.conf * 100)) + "%";
            cv::putText(view, label, cv::Point(d.box.x, std::max(15, d.box.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            cv::circle(view, foot, 4, color, -1);
            if (danger) alerts++;
        }

        if (alerts > 0) {
            cv::rectangle(view, cv::Point(0, 0), cv::Point(W - 1, H - 1), cv::Scalar(0, 0, 255), 6);
            cv::putText(view, "ALERT: object on track!", cv::Point(10, 55),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            if (savedAlerts < 5 && n % 10 == 0) {
                cv::imwrite("../alert_" + std::to_string(n) + ".png", view);
                savedAlerts++;
            }
        }

        cv::putText(view, init ? "Hazard zone: ON" : "Hazard zone: searching...",
                    cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        cv::imshow("1 Rail Perception", view);

        if (n % 50 == 0) {
            std::cout << "Frame " << n << ": detections " << dets.size()
                      << ", alerts " << alerts
                      << ", zone " << (init ? "ok" : "none") << std::endl;
        }

        int key = cv::waitKey(1);
        if (key == 'p') key = cv::waitKey(0);   // p = pause, next key = resume
        if (key == 27 || key == 'q') break;     // Esc or q = quit
        n++;
    }
    return 0;
}