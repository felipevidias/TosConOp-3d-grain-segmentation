#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tree_of_shapes.h"
#include "tree_of_shapes_edit.h"
#include "ttos_grain_params.h"
#include "ttos_grain_pipeline.h"
#include "ttos_grain_qc.h"

namespace
{
namespace fs = std::filesystem;

struct CropChoice {
    int x0 = 0;
    int y0 = 0;
    int z0 = 0;
    uint64_t nonzero_voxels = 0;
    double mean_nonzero = 0.0;
    bool valid = false;
};

double nonzero_percentile_from_volume(const std::vector<cv::Mat>& volume, double percentile) {
    std::array<uint64_t, 256> hist{};
    hist.fill(0);
    uint64_t total_nonzero = 0;

    for (const auto& slice : volume) {
        for (int y = 0; y < slice.rows; ++y) {
            const uchar* row = slice.ptr<uchar>(y);
            for (int x = 0; x < slice.cols; ++x) {
                const uchar v = row[x];
                if (v > 0) {
                    hist[v]++;
                    total_nonzero++;
                }
            }
        }
    }

    if (total_nonzero == 0) return 0.0;

    percentile = std::clamp(percentile, 0.0, 100.0);
    const uint64_t target =
        static_cast<uint64_t>(std::ceil((percentile / 100.0) * static_cast<double>(total_nonzero)));

    uint64_t acc = 0;
    for (int v = 1; v < 256; ++v) {
        acc += hist[v];
        if (acc >= target) return static_cast<double>(v);
    }
    return 255.0;
}

std::vector<cv::Mat> extract_crop_zyx(
    const std::vector<cv::Mat>& volume,
    int x0,
    int y0,
    int z0,
    int crop_w,
    int crop_h,
    int crop_d)
{
    std::vector<cv::Mat> out;
    out.reserve(crop_d);
    const cv::Rect roi(x0, y0, crop_w, crop_h);
    for (int z = z0; z < z0 + crop_d; ++z) {
        out.push_back(volume[z](roi).clone());
    }
    return out;
}

std::pair<uint64_t, double> crop_stats(
    const std::vector<cv::Mat>& volume,
    int x0,
    int y0,
    int z0,
    int crop_w,
    int crop_h,
    int crop_d)
{
    uint64_t nnz = 0;
    uint64_t sum = 0;

    for (int z = z0; z < z0 + crop_d; ++z) {
        cv::Mat roi = volume[z](cv::Rect(x0, y0, crop_w, crop_h));
        for (int y = 0; y < roi.rows; ++y) {
            const uchar* row = roi.ptr<uchar>(y);
            for (int x = 0; x < roi.cols; ++x) {
                const uchar v = row[x];
                if (v > 0) {
                    nnz++;
                    sum += v;
                }
            }
        }
    }

    const double mean_nonzero =
        nnz > 0 ? static_cast<double>(sum) / static_cast<double>(nnz) : 0.0;
    return {nnz, mean_nonzero};
}

CropChoice find_best_offcenter_crop(
    const std::vector<cv::Mat>& volume,
    int crop_w,
    int crop_h,
    int crop_d)
{
    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    const int center_x0 = std::max(0, (width - crop_w) / 2);
    const int center_y0 = std::max(0, (height - crop_h) / 2);
    const int center_z0 = std::max(0, (depth - crop_d) / 2);

    const int step_x = std::max(32, crop_w / 2);
    const int step_y = std::max(32, crop_h / 2);
    const int step_z = std::max(32, crop_d / 2);

    CropChoice best;

    for (int z0 = 0; z0 <= depth - crop_d; z0 += step_z) {
        for (int y0 = 0; y0 <= height - crop_h; y0 += step_y) {
            for (int x0 = 0; x0 <= width - crop_w; x0 += step_x) {
                const bool near_center =
                    std::abs(x0 - center_x0) < crop_w / 2 &&
                    std::abs(y0 - center_y0) < crop_h / 2 &&
                    std::abs(z0 - center_z0) < crop_d / 2;

                if (near_center) continue;

                auto [nnz, mean_nonzero] =
                    crop_stats(volume, x0, y0, z0, crop_w, crop_h, crop_d);

                if (!best.valid ||
                    nnz > best.nonzero_voxels ||
                    (nnz == best.nonzero_voxels && mean_nonzero > best.mean_nonzero)) {
                    best = {x0, y0, z0, nnz, mean_nonzero, true};
                }
            }
        }
    }

    if (!best.valid) {
        best.x0 = std::max(0, width / 6);
        best.y0 = std::max(0, height / 5);
        best.z0 = std::max(0, depth / 6);
        auto [nnz, mean_nonzero] =
            crop_stats(volume, best.x0, best.y0, best.z0, crop_w, crop_h, crop_d);
        best.nonzero_voxels = nnz;
        best.mean_nonzero = mean_nonzero;
        best.valid = true;
    }

    return best;
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string out_path(const fs::path& results_dir, const std::string& name) {
    return (results_dir / name).string();
}

} // namespace

int main(int argc, char** argv) {
    TtosGrainParams params;
    const std::string input_tif =
        (argc >= 2) ? std::string(argv[1]) : params.default_input_tif;

    const int crop_w = 200;
    const int crop_h = 200;
    const int crop_d = 200;

    std::vector<cv::Mat> volume;
    cv::imreadmulti(input_tif, volume, cv::IMREAD_GRAYSCALE);
    if (volume.empty()) {
        std::cerr << "Error: could not read the TIFF volume." << std::endl;
        return 1;
    }

    const int depth = static_cast<int>(volume.size());
    const int height = volume[0].rows;
    const int width = volume[0].cols;

    int x0 = 0;
    int y0 = 0;
    int z0 = 0;

    if (argc >= 5) {
        x0 = std::stoi(argv[2]);
        y0 = std::stoi(argv[3]);
        z0 = std::stoi(argv[4]);
    } else {
        CropChoice best = find_best_offcenter_crop(volume, crop_w, crop_h, crop_d);
        x0 = best.x0;
        y0 = best.y0;
        z0 = best.z0;
        std::cout << "[1/6] Automatic off-center crop search" << std::endl;
        std::cout << "      chosen x0=" << x0 << ", y0=" << y0 << ", z0=" << z0 << std::endl;
        std::cout << "      nonzero voxels = " << best.nonzero_voxels << std::endl;
        std::cout << "      mean nonzero   = " << best.mean_nonzero << std::endl;
    }

    if (x0 + crop_w > width) x0 = std::max(0, width - crop_w);
    if (y0 + crop_h > height) y0 = std::max(0, height - crop_h);
    if (z0 + crop_d > depth) z0 = std::max(0, depth - crop_d);

    auto crop_volume = extract_crop_zyx(volume, x0, y0, z0, crop_w, crop_h, crop_d);
    const double p30_original = nonzero_percentile_from_volume(crop_volume, 30.0);
    const double p50_original = nonzero_percentile_from_volume(crop_volume, 50.0);
    const double p60_original = nonzero_percentile_from_volume(crop_volume, 60.0);

    std::cout << "Input volume: " << input_tif << std::endl;
    std::cout << "Full volume : " << width << "x" << height << "x" << depth << std::endl;
    std::cout << "Crop origin : x=" << x0 << ", y=" << y0 << ", z=" << z0 << std::endl;
    std::cout << "Crop size   : " << crop_w << "x" << crop_h << "x" << crop_d << std::endl;
    std::cout << "Crop nonzero percentiles:" << std::endl;
    std::cout << "  p30 = " << p30_original << std::endl;
    std::cout << "  p50 = " << p50_original << std::endl;
    std::cout << "  p60 = " << p60_original << std::endl;
    std::cout << "  dark_leaf_area_max   = " << params.dark_leaf_area_max << std::endl;
    std::cout << "  bright_leaf_area_max = " << params.bright_leaf_area_max << std::endl;

    if (p60_original <= 1.0) {
        std::cerr << "Error: selected crop is almost empty/background-only." << std::endl;
        return 1;
    }

    const auto t_global = std::chrono::steady_clock::now();

    fs::path results_dir = "results";
    fs::create_directories(results_dir);

    std::cout << "[2/6] Building one Tree of Shapes..." << std::endl;
    const auto t_tos = std::chrono::steady_clock::now();
    Tree_of_shapes tos_filtered(crop_volume);
    std::cout << "      tree nodes = " << tos_filtered.nb_nodes() << std::endl;
    std::cout << "      done in " << std::fixed << std::setprecision(2)
              << seconds_since(t_tos) << " s" << std::endl;

    std::cout << "[3/6] Running selective ToSConOp passes..." << std::endl;
    const auto t_filter = std::chrono::steady_clock::now();

    ToSLeafFilterParams dark_pass;
    dark_pass.max_area = params.dark_leaf_area_max;
    dark_pass.max_delta_parent = params.dark_max_delta_parent;
    dark_pass.max_interval_amplitude = params.dark_max_interval_amplitude;
    dark_pass.node_class_constraint = MIN_TREE;
    dark_pass.progress_label = "Dark pass";

    ToSLeafFilterParams bright_pass;
    bright_pass.max_area = params.bright_leaf_area_max;
    bright_pass.max_delta_parent = params.bright_max_delta_parent;
    bright_pass.max_interval_amplitude = params.bright_max_interval_amplitude;
    bright_pass.node_class_constraint = MAX_TREE;
    bright_pass.progress_label = "Bright pass";

    const ToSPaperOperatorStats dark_stats =
        apply_selective_leaf_filter(
            tos_filtered,
            dark_pass,
            params.show_operator_progress,
            params.operator_progress_every,
            params.operator_refresh_every);

    const ToSPaperOperatorStats bright_stats =
        apply_selective_leaf_filter(
            tos_filtered,
            bright_pass,
            params.show_operator_progress,
            params.operator_progress_every,
            params.operator_refresh_every);

    auto filtered_volume = tos_filtered.reconstruct_image_3d();
    const double p30_filtered = nonzero_percentile_from_volume(filtered_volume, 30.0);
    const double p50_filtered = nonzero_percentile_from_volume(filtered_volume, 50.0);
    const double p60_filtered = nonzero_percentile_from_volume(filtered_volume, 60.0);

    std::cout << "      done in " << std::fixed << std::setprecision(2)
              << seconds_since(t_filter) << " s" << std::endl;

    std::cout << "[4/6] Extracting tree-native supports..." << std::endl;
    const auto t_support = std::chrono::steady_clock::now();
    auto support_candidates =
        extract_support_candidates(tos_filtered, filtered_volume, params, p30_filtered, p60_filtered);
    auto final_supports = nms_support_candidates(support_candidates, params);
    auto grains = make_final_grains_from_supports(final_supports, tos_filtered, filtered_volume, params);

    auto support_labels =
        build_support_label_volume(grains, crop_volume[0].cols, crop_volume[0].rows, static_cast<int>(crop_volume.size()));

    std::cout << "      support candidates = " << support_candidates.size() << std::endl;
    std::cout << "      final supports     = " << final_supports.size() << std::endl;
    std::cout << "      grains             = " << grains.size() << std::endl;
    std::cout << "      done in " << std::fixed << std::setprecision(2)
              << seconds_since(t_support) << " s" << std::endl;

    std::cout << "[5/6] Extracting seeds from MIN_TREE descendants..." << std::endl;
    const auto t_seed = std::chrono::steady_clock::now();
    assign_dark_ttos_seeds(tos_filtered, filtered_volume, params, p50_filtered, grains);

    auto markers =
        build_marker_volume(grains, crop_volume[0].cols, crop_volume[0].rows, static_cast<int>(crop_volume.size()));
    auto final_labels =
        build_final_grain_label_volume(grains, crop_volume[0].cols, crop_volume[0].rows, static_cast<int>(crop_volume.size()));

    std::size_t grains_with_seed = 0;
    for (const auto& g : grains) {
        if (g.seed_x >= 0 && g.seed_y >= 0 && g.seed_z >= 0) grains_with_seed++;
    }

    std::cout << "      grains with seeds  = " << grains_with_seed << " / " << grains.size() << std::endl;
    std::cout << "      done in " << std::fixed << std::setprecision(2)
              << seconds_since(t_seed) << " s" << std::endl;

    std::cout << "[6/6] Writing outputs..." << std::endl;
    cv::imwritemulti(out_path(results_dir, "input_gray_stack_8u.tif"), crop_volume);
    cv::imwritemulti(out_path(results_dir, "filtered_volume_8u.tif"), filtered_volume);
    cv::imwritemulti(out_path(results_dir, "support_labels_16u.tif"), support_labels);
    cv::imwritemulti(out_path(results_dir, "grain_markers_points_16u.tif"), markers);
    cv::imwritemulti(out_path(results_dir, "final_grain_labels_16u.tif"), final_labels);

    std::vector<cv::Mat> track_qc_rgb;
    build_persistent_track_qc(crop_volume, grains, track_qc_rgb);
    cv::imwritemulti(out_path(results_dir, "grain_tracks_qc_rgb.tif"), track_qc_rgb);

    cv::Mat seed_ortho = render_seed_orthoview_qc(crop_volume, grains);
    cv::imwrite(out_path(results_dir, "seed_orthoview_qc.png"), seed_ortho);

    save_seed_csv(out_path(results_dir, "seed_list.csv"), grains);

    std::ofstream meta(out_path(results_dir, "metadata.txt"));
    meta << "input=" << input_tif << "\n";
    meta << "full_shape=" << width << "x" << height << "x" << depth << "\n";
    meta << "crop_origin_x=" << x0 << "\n";
    meta << "crop_origin_y=" << y0 << "\n";
    meta << "crop_origin_z=" << z0 << "\n";
    meta << "crop_shape=" << crop_w << "x" << crop_h << "x" << crop_d << "\n";
    meta << "dark_leaf_area_max=" << params.dark_leaf_area_max << "\n";
    meta << "bright_leaf_area_max=" << params.bright_leaf_area_max << "\n";
    meta << "dark_max_delta_parent=" << params.dark_max_delta_parent << "\n";
    meta << "bright_max_delta_parent=" << params.bright_max_delta_parent << "\n";
    meta << "dark_max_interval_amplitude=" << params.dark_max_interval_amplitude << "\n";
    meta << "bright_max_interval_amplitude=" << params.bright_max_interval_amplitude << "\n";
    meta << "p30_original=" << p30_original << "\n";
    meta << "p50_original=" << p50_original << "\n";
    meta << "p60_original=" << p60_original << "\n";
    meta << "p30_filtered=" << p30_filtered << "\n";
    meta << "p50_filtered=" << p50_filtered << "\n";
    meta << "p60_filtered=" << p60_filtered << "\n";
    meta << "support_candidates=" << support_candidates.size() << "\n";
    meta << "final_supports=" << final_supports.size() << "\n";
    meta << "grains=" << grains.size() << "\n";
    meta << "grains_with_seed=" << grains_with_seed << "\n";
    meta << "dark_iterations=" << dark_stats.iterations << "\n";
    meta << "dark_discarded_nodes=" << dark_stats.discarded_nodes << "\n";
    meta << "dark_shifted_nodes=" << dark_stats.shifted_nodes << "\n";
    meta << "dark_initial_candidate_leaves=" << dark_stats.initial_candidate_leaves << "\n";
    meta << "bright_iterations=" << bright_stats.iterations << "\n";
    meta << "bright_discarded_nodes=" << bright_stats.discarded_nodes << "\n";
    meta << "bright_shifted_nodes=" << bright_stats.shifted_nodes << "\n";
    meta << "bright_initial_candidate_leaves=" << bright_stats.initial_candidate_leaves << "\n";
    meta << "total_runtime_seconds=" << seconds_since(t_global) << "\n";

    std::cout << "      done." << std::endl;
    std::cout << "\nGenerated files in: " << results_dir.string() << std::endl;
    std::cout << "  input_gray_stack_8u.tif" << std::endl;
    std::cout << "  filtered_volume_8u.tif" << std::endl;
    std::cout << "  support_labels_16u.tif" << std::endl;
    std::cout << "  grain_markers_points_16u.tif" << std::endl;
    std::cout << "  final_grain_labels_16u.tif" << std::endl;
    std::cout << "  grain_tracks_qc_rgb.tif" << std::endl;
    std::cout << "  seed_orthoview_qc.png" << std::endl;
    std::cout << "  seed_list.csv" << std::endl;
    std::cout << "  metadata.txt" << std::endl;
    std::cout << "\nTotal runtime: " << std::fixed << std::setprecision(2)
              << seconds_since(t_global) << " s" << std::endl;
    return 0;
}
