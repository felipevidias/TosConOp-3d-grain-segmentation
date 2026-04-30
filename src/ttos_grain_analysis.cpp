#include "ttos_grain_analysis.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

double clamp01(double v)
{
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

double distance3d(
    double x1, double y1, double z1,
    double x2, double y2, double z2)
{
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    const double dz = z1 - z2;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool is_ancestor(Node_tos* possible_ancestor, Node_tos* node)
{
    if (possible_ancestor == nullptr || node == nullptr) {
        return false;
    }

    Node_tos* cur = node->parent;
    while (cur != nullptr) {
        if (cur == possible_ancestor) {
            return true;
        }
        cur = cur->parent;
    }
    return false;
}

void decode_voxel(long p, int width, int height, int& x, int& y, int& z)
{
    const long slice_area = static_cast<long>(width) * height;
    z = static_cast<int>(p / slice_area);
    const long rem = p % slice_area;
    y = static_cast<int>(rem / width);
    x = static_cast<int>(rem % width);
}

double percentile_from_volume(const std::vector<cv::Mat>& volume, double p)
{
    std::vector<uchar> values;
    values.reserve(static_cast<std::size_t>(volume.size()) * volume[0].rows * volume[0].cols);

    for (const auto& slice : volume) {
        for (int y = 0; y < slice.rows; ++y) {
            const uchar* row = slice.ptr<uchar>(y);
            for (int x = 0; x < slice.cols; ++x) {
                values.push_back(row[x]);
            }
        }
    }

    if (values.empty()) {
        return 0.0;
    }

    p = std::max(0.0, std::min(100.0, p));
    const std::size_t idx = static_cast<std::size_t>(
        std::round((p / 100.0) * (values.size() - 1)));

    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return static_cast<double>(values[idx]);
}

void collect_subtree_voxels(const Node_tos* node, std::vector<long>& voxels)
{
    std::vector<const Node_tos*> stack;
    stack.push_back(node);

    while (!stack.empty()) {
        const Node_tos* cur = stack.back();
        stack.pop_back();

        for (long p : cur->proper_part) {
            voxels.push_back(p);
        }

        for (const Node_tos* child : cur->children) {
            if (child != nullptr && !child->removed) {
                stack.push_back(child);
            }
        }
    }
}

GeoStats compute_geometry(
    const std::vector<long>& voxels,
    int width,
    int height,
    int depth)
{
    GeoStats g;
    g.volume = static_cast<long>(voxels.size());

    if (voxels.empty()) {
        return g;
    }

    std::vector<int> has_slice(depth, 0);

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    long valid = 0;

    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        g.minx = std::min(g.minx, x);
        g.miny = std::min(g.miny, y);
        g.minz = std::min(g.minz, z);

        g.maxx = std::max(g.maxx, x);
        g.maxy = std::max(g.maxy, y);
        g.maxz = std::max(g.maxz, z);

        sx += x;
        sy += y;
        sz += z;
        valid++;
        has_slice[z] = 1;

        if (x == 0 || y == 0 || z == 0 ||
            x == width - 1 || y == height - 1 || z == depth - 1) {
            g.touches_border = true;
        }
    }

    if (valid == 0) {
        return g;
    }

    g.dx = g.maxx - g.minx + 1;
    g.dy = g.maxy - g.miny + 1;
    g.dz = g.maxz - g.minz + 1;

    g.cx = sx / static_cast<double>(valid);
    g.cy = sy / static_cast<double>(valid);
    g.cz = sz / static_cast<double>(valid);

    for (int z = 0; z < depth; ++z) {
        if (has_slice[z]) {
            g.num_slices++;
        }
    }

    return g;
}

double compute_mean_intensity(
    const std::vector<long>& voxels,
    const std::vector<cv::Mat>& volume,
    int width,
    int height,
    int depth)
{
    if (voxels.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    long count = 0;

    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        sum += volume[z].at<uchar>(y, x);
        count++;
    }

    if (count == 0) {
        return 0.0;
    }

    return sum / static_cast<double>(count);
}

double bbox_overlap_ratio_min(const GeoStats& a, const GeoStats& b)
{
    const int ix0 = std::max(a.minx, b.minx);
    const int iy0 = std::max(a.miny, b.miny);
    const int iz0 = std::max(a.minz, b.minz);

    const int ix1 = std::min(a.maxx, b.maxx);
    const int iy1 = std::min(a.maxy, b.maxy);
    const int iz1 = std::min(a.maxz, b.maxz);

    const int dx = ix1 - ix0 + 1;
    const int dy = iy1 - iy0 + 1;
    const int dz = iz1 - iz0 + 1;

    if (dx <= 0 || dy <= 0 || dz <= 0) {
        return 0.0;
    }

    const double inter = static_cast<double>(dx) * dy * dz;
    const double va = static_cast<double>(std::max(1, a.dx)) *
                      std::max(1, a.dy) *
                      std::max(1, a.dz);
    const double vb = static_cast<double>(std::max(1, b.dx)) *
                      std::max(1, b.dy) *
                      std::max(1, b.dz);

    const double denom = std::max(1.0, std::min(va, vb));
    return inter / denom;
}

ZTrackStats compute_ztrack(
    const std::vector<long>& voxels,
    int width,
    int height,
    int depth)
{
    ZTrackStats t;

    std::vector<int> area(depth, 0);
    std::vector<double> sx(depth, 0.0);
    std::vector<double> sy(depth, 0.0);

    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        area[z]++;
        sx[z] += x;
        sy[z] += y;
    }

    int peak = 0;
    int z_peak = 0;
    for (int z = 0; z < depth; ++z) {
        if (area[z] > peak) {
            peak = area[z];
            z_peak = z;
        }
    }

    t.area_peak = peak;
    t.z_peak = z_peak;

    if (peak <= 0) {
        return t;
    }

    const int active_thr = std::max(2, static_cast<int>(std::round(0.03 * peak)));

    t.z_min_active = depth - 1;
    t.z_max_active = 0;

    int current_run = 0;
    int longest_run = 0;

    for (int z = 0; z < depth; ++z) {
        const bool active = area[z] >= active_thr;
        if (active) {
            t.active_slices++;
            t.z_min_active = std::min(t.z_min_active, z);
            t.z_max_active = std::max(t.z_max_active, z);
            current_run++;
            longest_run = std::max(longest_run, current_run);
        } else {
            current_run = 0;
        }
    }

    t.longest_run = longest_run;

    const int span = std::max(1, t.z_max_active - t.z_min_active + 1);
    t.continuity_score = clamp01(static_cast<double>(t.longest_run) / span);

    double area_variation = 0.0;
    double centroid_jump_sum = 0.0;
    int area_pairs = 0;
    int centroid_pairs = 0;

    int prev_area = -1;
    double prev_cx = 0.0;
    double prev_cy = 0.0;
    bool prev_centroid_valid = false;

    for (int z = t.z_min_active; z <= t.z_max_active; ++z) {
        if (area[z] < active_thr) {
            continue;
        }

        if (prev_area >= 0) {
            const double denom = static_cast<double>(std::max(1, std::max(prev_area, area[z])));
            area_variation += std::abs(static_cast<double>(area[z] - prev_area)) / denom;
            area_pairs++;
        }

        const double cx = sx[z] / static_cast<double>(area[z]);
        const double cy = sy[z] / static_cast<double>(area[z]);

        if (prev_centroid_valid) {
            centroid_jump_sum += distance3d(cx, cy, 0.0, prev_cx, prev_cy, 0.0);
            centroid_pairs++;
        }

        prev_area = area[z];
        prev_cx = cx;
        prev_cy = cy;
        prev_centroid_valid = true;
    }

    const double mean_area_var =
        (area_pairs > 0) ? area_variation / static_cast<double>(area_pairs) : 0.0;

    const double mean_centroid_jump =
        (centroid_pairs > 0) ? centroid_jump_sum / static_cast<double>(centroid_pairs) : 0.0;

    t.area_smooth_score = clamp01(1.0 - mean_area_var);
    t.centroid_smooth_score = clamp01(1.0 - mean_centroid_jump / 8.0);

    t.track_score =
        0.45 * t.continuity_score +
        0.30 * t.area_smooth_score +
        0.25 * t.centroid_smooth_score;

    return t;
}

LocalStats compute_local_stats(
    const std::vector<cv::Mat>& volume,
    int cx,
    int cy,
    int cz)
{
    LocalStats s;

    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    const int center_radius = 1;
    const int shell_inner = 4;
    const int shell_outer = 10;

    const int center_r2 = center_radius * center_radius;
    const int shell_inner2 = shell_inner * shell_inner;
    const int shell_outer2 = shell_outer * shell_outer;

    double center_sum = 0.0;
    double shell_sum = 0.0;
    long center_count = 0;
    long shell_count = 0;

    for (int dz = -shell_outer; dz <= shell_outer; ++dz) {
        for (int dy = -shell_outer; dy <= shell_outer; ++dy) {
            for (int dx = -shell_outer; dx <= shell_outer; ++dx) {
                const int d2 = dx * dx + dy * dy + dz * dz;

                const int z = cz + dz;
                const int y = cy + dy;
                const int x = cx + dx;

                if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
                    continue;
                }

                const uchar v = volume[z].at<uchar>(y, x);

                if (d2 <= center_r2) {
                    center_sum += v;
                    center_count++;
                }

                if (d2 >= shell_inner2 && d2 <= shell_outer2) {
                    shell_sum += v;
                    shell_count++;
                }
            }
        }
    }

    if (center_count > 0) {
        s.center_mean = center_sum / static_cast<double>(center_count);
    }

    if (shell_count > 0) {
        s.shell_mean = shell_sum / static_cast<double>(shell_count);
    }

    s.contrast = s.shell_mean - s.center_mean;
    return s;
}

bool find_representative_voxel_of_node(
    const std::vector<long>& voxels,
    const std::vector<cv::Mat>& volume,
    const GeoStats& geo,
    int width,
    int height,
    int depth,
    int& out_x,
    int& out_y,
    int& out_z,
    int& out_value)
{
    if (voxels.empty()) {
        return false;
    }

    bool found = false;
    double best_score = std::numeric_limits<double>::infinity();

    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        const bool internal =
            x > geo.minx + 1 && x < geo.maxx - 1 &&
            y > geo.miny + 1 && y < geo.maxy - 1 &&
            z > geo.minz + 1 && z < geo.maxz - 1;

        if ((geo.dx > 5 && geo.dy > 5 && geo.dz > 5) && !internal) {
            continue;
        }

        const int value = static_cast<int>(volume[z].at<uchar>(y, x));
        const double d_center = distance3d(
            static_cast<double>(x), static_cast<double>(y), static_cast<double>(z),
            geo.cx, geo.cy, geo.cz);

        const double score = static_cast<double>(value) + 0.20 * d_center;

        if (score < best_score) {
            best_score = score;
            out_x = x;
            out_y = y;
            out_z = z;
            out_value = value;
            found = true;
        }
    }

    if (found) {
        return true;
    }

    best_score = std::numeric_limits<double>::infinity();
    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        const double d_center = distance3d(
            static_cast<double>(x), static_cast<double>(y), static_cast<double>(z),
            geo.cx, geo.cy, geo.cz);

        if (d_center < best_score) {
            best_score = d_center;
            out_x = x;
            out_y = y;
            out_z = z;
            out_value = static_cast<int>(volume[z].at<uchar>(y, x));
            found = true;
        }
    }

    return found;
}

bool choose_fallback_seed_from_support(
    const std::vector<long>& voxels,
    const GeoStats& geo,
    int width,
    int height,
    int depth,
    int& sx,
    int& sy,
    int& sz)
{
    if (voxels.empty()) {
        return false;
    }

    double best_score = std::numeric_limits<double>::infinity();
    bool found = false;

    for (long p : voxels) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_voxel(p, width, height, x, y, z);

        if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
            continue;
        }

        const double d = distance3d(
            static_cast<double>(x), static_cast<double>(y), static_cast<double>(z),
            geo.cx, geo.cy, geo.cz);

        if (d < best_score) {
            best_score = d;
            sx = x;
            sy = y;
            sz = z;
            found = true;
        }
    }

    return found;
}

void trim_support_candidates(
    std::vector<SupportCandidate>& candidates,
    std::size_t max_keep)
{
    if (candidates.size() <= max_keep) {
        return;
    }

    std::nth_element(
        candidates.begin(),
        candidates.begin() + static_cast<long>(max_keep),
        candidates.end(),
        [](const SupportCandidate& a, const SupportCandidate& b) {
            return a.score > b.score;
        });

    candidates.resize(max_keep);
}
