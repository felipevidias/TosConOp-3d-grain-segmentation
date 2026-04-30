#include "ttos_grain_qc.h"

#include <algorithm>
#include <string>

#include "ttos_grain_analysis.h"

cv::Vec3b color_from_label(int label)
{
    cv::Mat hsv(1, 1, CV_8UC3);
    const int hue = (label * 37) % 180;
    hsv.at<cv::Vec3b>(0, 0) = cv::Vec3b((uchar)hue, (uchar)255, (uchar)255);

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

void paint_seed_qc(
    std::vector<cv::Mat>& qc,
    int x,
    int y,
    int z,
    int label,
    const cv::Vec3b& color)
{
    if (z < 0 || z >= static_cast<int>(qc.size())) {
        return;
    }

    const int h = qc[z].rows;
    const int w = qc[z].cols;
    const int radius = 2;
    const int r2 = radius * radius;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > r2) {
                continue;
            }

            const int xx = x + dx;
            const int yy = y + dy;

            if (xx < 0 || xx >= w || yy < 0 || yy >= h) {
                continue;
            }

            qc[z].at<cv::Vec3b>(yy, xx) = color;
        }
    }

    cv::putText(
        qc[z],
        std::to_string(label),
        cv::Point(x + 3, y - 3),
        cv::FONT_HERSHEY_SIMPLEX,
        0.28,
        cv::Scalar(color[0], color[1], color[2]),
        1,
        cv::LINE_AA);
}

void build_persistent_track_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains,
    std::vector<cv::Mat>& out_rgb)
{
    out_rgb.clear();
    out_rgb.reserve(volume.size());

    for (const cv::Mat& slice : volume) {
        cv::Mat rgb;
        cv::cvtColor(slice, rgb, cv::COLOR_GRAY2BGR);
        out_rgb.push_back(rgb);
    }

    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    for (const FinalGrain& g : grains) {
        std::vector<long> count(depth, 0);
        std::vector<long> sx(depth, 0);
        std::vector<long> sy(depth, 0);

        for (long p : g.voxels) {
            int x = 0, y = 0, z = 0;
            decode_voxel(p, width, height, x, y, z);

            if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
                continue;
            }

            count[z] += 1;
            sx[z] += x;
            sy[z] += y;
        }

        const cv::Vec3b color = color_from_label(g.label);

        for (int z = 0; z < depth; ++z) {
            if (count[z] <= 0) {
                continue;
            }

            int cx = static_cast<int>(std::round(static_cast<double>(sx[z]) / count[z]));
            int cy = static_cast<int>(std::round(static_cast<double>(sy[z]) / count[z]));

            cx = std::max(0, std::min(width - 1, cx));
            cy = std::max(0, std::min(height - 1, cy));

            cv::putText(
                out_rgb[z],
                std::to_string(g.label),
                cv::Point(cx + 2, cy - 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.28,
                cv::Scalar(color[0], color[1], color[2]),
                1,
                cv::LINE_AA);

            if (z == g.seed_z && g.seed_x >= 0 && g.seed_y >= 0) {
                const int radius = 2;
                const int r2 = radius * radius;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (dx * dx + dy * dy > r2) {
                            continue;
                        }

                        const int xx = g.seed_x + dx;
                        const int yy = g.seed_y + dy;

                        if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
                            continue;
                        }

                        out_rgb[z].at<cv::Vec3b>(yy, xx) = color;
                    }
                }
            }
        }
    }
}

static cv::Mat make_mip_xy(const std::vector<cv::Mat>& volume)
{
    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            const uchar* srcp = volume[z].ptr<uchar>(y);
            uchar* dstp = out.ptr<uchar>(y);

            for (int x = 0; x < width; ++x) {
                if (srcp[x] > dstp[x]) {
                    dstp[x] = srcp[x];
                }
            }
        }
    }

    return out;
}

static cv::Mat make_mip_xz(const std::vector<cv::Mat>& volume)
{
    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    cv::Mat out = cv::Mat::zeros(depth, width, CV_8UC1);

    for (int z = 0; z < depth; ++z) {
        uchar* dst_row = out.ptr<uchar>(z);

        for (int y = 0; y < height; ++y) {
            const uchar* srcp = volume[z].ptr<uchar>(y);

            for (int x = 0; x < width; ++x) {
                if (srcp[x] > dst_row[x]) {
                    dst_row[x] = srcp[x];
                }
            }
        }
    }

    return out;
}

static cv::Mat make_mip_yz(const std::vector<cv::Mat>& volume)
{
    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    cv::Mat out = cv::Mat::zeros(depth, height, CV_8UC1);

    for (int z = 0; z < depth; ++z) {
        uchar* dst_row = out.ptr<uchar>(z);

        for (int y = 0; y < height; ++y) {
            const uchar* srcp = volume[z].ptr<uchar>(y);
            uchar vmax = 0;

            for (int x = 0; x < width; ++x) {
                if (srcp[x] > vmax) {
                    vmax = srcp[x];
                }
            }

            if (vmax > dst_row[y]) {
                dst_row[y] = vmax;
            }
        }
    }

    return out;
}

static void draw_seed_dot(cv::Mat& img, int u, int v, const cv::Vec3b& color)
{
    const int radius = 2;
    const int r2 = radius * radius;

    for (int dv = -radius; dv <= radius; ++dv) {
        for (int du = -radius; du <= radius; ++du) {
            if (du * du + dv * dv > r2) {
                continue;
            }

            const int uu = u + du;
            const int vv = v + dv;

            if (uu < 0 || uu >= img.cols || vv < 0 || vv >= img.rows) {
                continue;
            }

            img.at<cv::Vec3b>(vv, uu) = color;
        }
    }
}

static void draw_crosshair(cv::Mat& img, int u, int v)
{
    cv::line(img, cv::Point(0, v), cv::Point(img.cols - 1, v),
             cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    cv::line(img, cv::Point(u, 0), cv::Point(u, img.rows - 1),
             cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
}

static cv::Mat assemble_ortho_canvas(
    const cv::Mat& xy_rgb,
    const cv::Mat& xz_rgb,
    const cv::Mat& yz_rgb)
{
    const int w = xy_rgb.cols;
    const int h = xy_rgb.rows;
    const int top_h = std::max(h, xz_rgb.rows);
    const int bot_h = yz_rgb.rows;
    const int right_w = std::max(xz_rgb.cols, yz_rgb.cols);

    cv::Mat canvas = cv::Mat::zeros(top_h + bot_h, w + right_w, CV_8UC3);
    canvas.setTo(cv::Scalar(30, 30, 30));

    xy_rgb.copyTo(canvas(cv::Rect(0, 0, xy_rgb.cols, xy_rgb.rows)));
    xz_rgb.copyTo(canvas(cv::Rect(w, 0, xz_rgb.cols, xz_rgb.rows)));
    yz_rgb.copyTo(canvas(cv::Rect(w, top_h, yz_rgb.cols, yz_rgb.rows)));

    cv::putText(canvas, "XY MIP", cv::Point(10, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(canvas, "XZ MIP", cv::Point(w + 10, 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(canvas, "YZ MIP", cv::Point(w + 10, top_h + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    return canvas;
}

cv::Mat render_seed_orthoview_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains)
{
    cv::Mat xy = make_mip_xy(volume);
    cv::Mat xz = make_mip_xz(volume);
    cv::Mat yz = make_mip_yz(volume);

    cv::Mat xy_rgb, xz_rgb, yz_rgb;
    cv::cvtColor(xy, xy_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(xz, xz_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(yz, yz_rgb, cv::COLOR_GRAY2BGR);

    for (const FinalGrain& g : grains) {
        if (g.seed_x < 0 || g.seed_y < 0 || g.seed_z < 0) {
            continue;
        }

        const cv::Vec3b color = color_from_label(g.label);

        draw_seed_dot(xy_rgb, g.seed_x, g.seed_y, color);
        draw_seed_dot(xz_rgb, g.seed_x, g.seed_z, color);
        draw_seed_dot(yz_rgb, g.seed_y, g.seed_z, color);

        cv::putText(xy_rgb, std::to_string(g.label),
                    cv::Point(g.seed_x + 4, std::max(10, g.seed_y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);

        cv::putText(xz_rgb, std::to_string(g.label),
                    cv::Point(g.seed_x + 4, std::max(10, g.seed_z - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);

        cv::putText(yz_rgb, std::to_string(g.label),
                    cv::Point(g.seed_y + 4, std::max(10, g.seed_z - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
    }

    return assemble_ortho_canvas(xy_rgb, xz_rgb, yz_rgb);
}

cv::Mat render_track_orthoview_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains)
{
    cv::Mat xy = make_mip_xy(volume);
    cv::Mat xz = make_mip_xz(volume);
    cv::Mat yz = make_mip_yz(volume);

    cv::Mat xy_rgb, xz_rgb, yz_rgb;
    cv::cvtColor(xy, xy_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(xz, xz_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(yz, yz_rgb, cv::COLOR_GRAY2BGR);

    for (const FinalGrain& g : grains) {
        const cv::Vec3b color = color_from_label(g.label);

        int cxy_x = static_cast<int>(std::round(g.geo.cx));
        int cxy_y = static_cast<int>(std::round(g.geo.cy));
        int cxz_x = static_cast<int>(std::round(g.geo.cx));
        int cxz_z = static_cast<int>(std::round(g.geo.cz));
        int cyz_y = static_cast<int>(std::round(g.geo.cy));
        int cyz_z = static_cast<int>(std::round(g.geo.cz));

        cxy_x = std::max(0, std::min(xy_rgb.cols - 1, cxy_x));
        cxy_y = std::max(0, std::min(xy_rgb.rows - 1, cxy_y));
        cxz_x = std::max(0, std::min(xz_rgb.cols - 1, cxz_x));
        cxz_z = std::max(0, std::min(xz_rgb.rows - 1, cxz_z));
        cyz_y = std::max(0, std::min(yz_rgb.cols - 1, cyz_y));
        cyz_z = std::max(0, std::min(yz_rgb.rows - 1, cyz_z));

        draw_seed_dot(xy_rgb, cxy_x, cxy_y, color);
        draw_seed_dot(xz_rgb, cxz_x, cxz_z, color);
        draw_seed_dot(yz_rgb, cyz_y, cyz_z, color);

        cv::putText(xy_rgb, std::to_string(g.label),
                    cv::Point(cxy_x + 4, std::max(10, cxy_y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);

        cv::putText(xz_rgb, std::to_string(g.label),
                    cv::Point(cxz_x + 4, std::max(10, cxz_z - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);

        cv::putText(yz_rgb, std::to_string(g.label),
                    cv::Point(cyz_y + 4, std::max(10, cyz_z - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35,
                    cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);

        if (g.seed_x >= 0 && g.seed_y >= 0 && g.seed_z >= 0) {
            draw_crosshair(xy_rgb, g.seed_x, g.seed_y);
            draw_crosshair(xz_rgb, g.seed_x, g.seed_z);
            draw_crosshair(yz_rgb, g.seed_y, g.seed_z);
        }
    }

    return assemble_ortho_canvas(xy_rgb, xz_rgb, yz_rgb);
}
