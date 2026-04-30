#include "ttos_grain_pipeline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "ttos_grain_analysis.h"

namespace
{
bool is_support_node(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) {
        return false;
    }

    return node->node_class == MAX_TREE && node->alt > node->parent->alt;
}

bool is_seed_node(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) {
        return false;
    }

    return node->node_class == MIN_TREE && node->alt < node->parent->alt;
}

double node_interval_amplitude(const Node_tos* node)
{
    if (node == nullptr) {
        return 0.0;
    }
    return static_cast<double>(std::abs(node->interval[1] - node->interval[0]));
}
}

std::vector<SupportCandidate> extract_support_candidates(
    Tree_of_shapes& tos_support,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams& params,
    double p30,
    double p60)
{
    std::vector<SupportCandidate> candidates;
    candidates.reserve(5000);

    tos_support.enrich();
    tos_support.compute_area();

    for (auto const& item : tos_support.nodes) {
        Node_tos* node = item.second.get();

        if (!is_support_node(node)) {
            continue;
        }

        if (node->area < params.support_area_min ||
            node->area > params.support_area_max) {
            continue;
        }

        std::vector<long> voxels;
        collect_subtree_voxels(node, voxels);
        if (voxels.empty()) {
            continue;
        }

        GeoStats geo = compute_geometry(
            voxels, tos_support.width, tos_support.height, tos_support.depth);

        if (geo.dx < params.support_dx_min || geo.dy < params.support_dy_min || geo.dz < params.support_dz_min ||
            geo.dx > params.support_dx_max || geo.dy > params.support_dy_max || geo.dz > params.support_dz_max) {
            continue;
        }

        if (geo.num_slices < params.min_active_slices) {
            continue;
        }

        ZTrackStats track = compute_ztrack(
            voxels, tos_support.width, tos_support.height, tos_support.depth);

        if (track.active_slices < params.min_active_slices ||
            track.longest_run < params.min_longest_z_run ||
            track.track_score < 0.35) {
            continue;
        }

        const double mean = compute_mean_intensity(
            voxels, volume, tos_support.width, tos_support.height, tos_support.depth);

        const double bbox_volume =
            static_cast<double>(std::max(1, geo.dx)) *
            std::max(1, geo.dy) *
            std::max(1, geo.dz);

        const double fill_ratio =
            bbox_volume > 0.0 ? static_cast<double>(geo.volume) / bbox_volume : 0.0;

        const double area_score = clamp01(
            (std::log(static_cast<double>(geo.volume) + 1.0) - std::log(120.0)) /
            (std::log(90000.0) - std::log(120.0)));

        const double brightness_score =
            clamp01((mean - p30) / std::max(1.0, p60 - p30));

        const double fill_score = clamp01(fill_ratio / 0.35);

        const int delta_parent = std::abs(static_cast<int>(node->parent->alt - node->alt));
        const double contrast_score =
            clamp01(static_cast<double>(delta_parent) / 70.0);

        const double interval_score =
            clamp01(node_interval_amplitude(node) / 70.0);

        const double border_penalty = geo.touches_border ? 0.35 : 0.0;

        double huge_penalty = 0.0;
        if (geo.dx > 80 || geo.dy > 80) huge_penalty += 0.20;
        if (geo.volume > 100000) huge_penalty += 0.25;

        SupportCandidate c;
        c.node = node;
        c.geo = geo;
        c.track = track;
        c.area = node->area;
        c.alt = node->alt;
        c.parent_alt = node->parent->alt;
        c.delta_parent = delta_parent;
        c.mean_intensity = mean;
        c.fill_ratio = fill_ratio;

        c.score =
            1.60 * track.track_score +
            1.15 * brightness_score +
            1.05 * fill_score +
            0.85 * area_score +
            0.55 * contrast_score +
            0.40 * interval_score -
            border_penalty -
            huge_penalty;

        if (c.score < params.support_score_min) {
            continue;
        }

        candidates.push_back(c);

        if (candidates.size() > params.max_support_candidates_keep * 2) {
            trim_support_candidates(candidates, params.max_support_candidates_keep);
        }
    }

    trim_support_candidates(candidates, params.max_support_candidates_keep);

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SupportCandidate& a, const SupportCandidate& b) {
            return a.score > b.score;
        });

    return candidates;
}

std::vector<SupportCandidate> nms_support_candidates(
    const std::vector<SupportCandidate>& candidates,
    const TtosGrainParams& params)
{
    std::vector<SupportCandidate> final_supports;
    final_supports.reserve(candidates.size());

    for (const SupportCandidate& cand : candidates) {
        if (final_supports.size() >= params.max_final_grains) {
            break;
        }

        bool duplicate = false;

        for (const SupportCandidate& acc : final_supports) {
            const double d = distance3d(
                cand.geo.cx, cand.geo.cy, cand.geo.cz,
                acc.geo.cx, acc.geo.cy, acc.geo.cz);

            if (d < params.support_center_nms) {
                duplicate = true;
                break;
            }

            const double overlap = bbox_overlap_ratio_min(cand.geo, acc.geo);
            if (overlap > params.support_bbox_overlap_nms) {
                duplicate = true;
                break;
            }

            if (d < 2.0 * params.support_center_nms) {
                const bool same_branch =
                    is_ancestor(cand.node, acc.node) ||
                    is_ancestor(acc.node, cand.node);

                if (same_branch) {
                    duplicate = true;
                    break;
                }
            }
        }

        if (!duplicate) {
            final_supports.push_back(cand);
        }
    }

    return final_supports;
}

std::vector<FinalGrain> make_final_grains_from_supports(
    const std::vector<SupportCandidate>& supports,
    Tree_of_shapes& tos_support)
{
    std::vector<FinalGrain> grains;
    grains.reserve(supports.size());

    int label = 1;
    for (const SupportCandidate& s : supports) {
        FinalGrain g;
        g.label = label++;
        g.node_id = s.node->name;
        g.geo = s.geo;
        g.track = s.track;
        g.area = s.area;
        g.alt = s.alt;
        g.parent_alt = s.parent_alt;
        g.delta_parent = s.delta_parent;
        g.mean_intensity = s.mean_intensity;
        g.fill_ratio = s.fill_ratio;
        g.score = s.score;

        collect_subtree_voxels(s.node, g.voxels);
        grains.push_back(std::move(g));
    }

    return grains;
}

std::vector<cv::Mat> build_support_label_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth)
{
    std::vector<cv::Mat> out;
    out.reserve(depth);

    for (int z = 0; z < depth; ++z) {
        out.push_back(cv::Mat::zeros(height, width, CV_16UC1));
    }

    for (const FinalGrain& g : grains) {
        const unsigned short lab = static_cast<unsigned short>(g.label);

        for (long p : g.voxels) {
            int x = 0, y = 0, z = 0;
            decode_voxel(p, width, height, x, y, z);

            if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
                continue;
            }

            if (out[z].at<unsigned short>(y, x) == 0) {
                out[z].at<unsigned short>(y, x) = lab;
            }
        }
    }

    return out;
}

void assign_dark_ttos_seeds(
    Tree_of_shapes& tos_seed,
    const std::vector<cv::Mat>& volume,
    const std::vector<cv::Mat>& support_labels,
    const TtosGrainParams& params,
    double p50,
    std::vector<FinalGrain>& grains)
{
    std::vector<double> best_seed_score(grains.size() + 1, -1e100);

    tos_seed.enrich();
    tos_seed.compute_area();

    for (auto const& item : tos_seed.nodes) {
        Node_tos* node = item.second.get();

        if (!is_seed_node(node)) {
            continue;
        }

        if (node->area < params.seed_area_min || node->area > params.seed_area_max) {
            continue;
        }

        const int delta = std::abs(static_cast<int>(node->parent->alt - node->alt));
        if (delta < params.seed_min_delta_parent) {
            continue;
        }

        std::vector<long> voxels;
        collect_subtree_voxels(node, voxels);

        if (voxels.empty()) {
            continue;
        }

        GeoStats geo = compute_geometry(
            voxels, tos_seed.width, tos_seed.height, tos_seed.depth);

        if (geo.dx > params.seed_dx_max ||
            geo.dy > params.seed_dy_max ||
            geo.dz > params.seed_dz_max) {
            continue;
        }

        int sx = 0, sy = 0, sz = 0, seed_value = 255;
        if (!find_representative_voxel_of_node(
                voxels,
                volume,
                geo,
                tos_seed.width,
                tos_seed.height,
                tos_seed.depth,
                sx, sy, sz, seed_value)) {
            continue;
        }

        if (sz < 0 || sz >= static_cast<int>(support_labels.size()) ||
            sy < 0 || sy >= support_labels[0].rows ||
            sx < 0 || sx >= support_labels[0].cols) {
            continue;
        }

        const unsigned short support_label = support_labels[sz].at<unsigned short>(sy, sx);
        if (support_label == 0) {
            continue;
        }

        const FinalGrain& support = grains[static_cast<int>(support_label) - 1];
        const LocalStats local = compute_local_stats(volume, sx, sy, sz);

        const double dark_score =
            clamp01((p50 - seed_value) / std::max(1.0, p50));

        const double contrast_score =
            clamp01(local.contrast / 60.0);

        const double delta_score =
            clamp01(static_cast<double>(delta) / 70.0);

        const double interval_score =
            clamp01(node_interval_amplitude(node) / 70.0);

        const double centrality =
            clamp01(
                1.0 - distance3d(
                    static_cast<double>(sx),
                    static_cast<double>(sy),
                    static_cast<double>(sz),
                    support.geo.cx,
                    support.geo.cy,
                    support.geo.cz) /
                std::max(
                    4.0,
                    0.5 * std::sqrt(
                        static_cast<double>(support.geo.dx * support.geo.dx +
                                            support.geo.dy * support.geo.dy +
                                            support.geo.dz * support.geo.dz))));

        double area_score = 1.0;
        if (node->area < 4) {
            area_score = clamp01(static_cast<double>(node->area) / 4.0);
        } else if (node->area > 1000) {
            area_score = clamp01(1.0 - (static_cast<double>(node->area) - 1000.0) / 4000.0);
        }

        const double score =
            1.90 * dark_score +
            1.35 * contrast_score +
            0.85 * delta_score +
            0.70 * interval_score +
            0.75 * centrality +
            0.30 * area_score;

        if (score > best_seed_score[static_cast<int>(support_label)]) {
            best_seed_score[static_cast<int>(support_label)] = score;

            FinalGrain& g = grains[static_cast<int>(support_label) - 1];
            g.seed_x = sx;
            g.seed_y = sy;
            g.seed_z = sz;
            g.seed_value = seed_value;
            g.seed_from_dark_candidate = true;
        }
    }
}

void assign_support_fallback_seeds(
    const std::vector<cv::Mat>& volume,
    std::vector<FinalGrain>& grains)
{
    const int width = volume[0].cols;
    const int height = volume[0].rows;
    const int depth = static_cast<int>(volume.size());

    for (FinalGrain& g : grains) {
        if (g.seed_x >= 0 && g.seed_y >= 0 && g.seed_z >= 0) {
            continue;
        }

        int sx = 0, sy = 0, sz = 0;
        if (!choose_fallback_seed_from_support(
                g.voxels,
                g.geo,
                width,
                height,
                depth,
                sx,
                sy,
                sz)) {
            continue;
        }

        g.seed_x = sx;
        g.seed_y = sy;
        g.seed_z = sz;
        g.seed_value = volume[sz].at<uchar>(sy, sx);
        g.seed_from_dark_candidate = false;
    }
}

std::vector<cv::Mat> build_marker_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth)
{
    std::vector<cv::Mat> out;
    out.reserve(depth);

    for (int z = 0; z < depth; ++z) {
        out.push_back(cv::Mat::zeros(height, width, CV_16UC1));
    }

    for (const FinalGrain& g : grains) {
        if (g.seed_x < 0 || g.seed_y < 0 || g.seed_z < 0) {
            continue;
        }

        if (g.seed_z >= 0 && g.seed_z < depth &&
            g.seed_y >= 0 && g.seed_y < height &&
            g.seed_x >= 0 && g.seed_x < width) {
            out[g.seed_z].at<unsigned short>(g.seed_y, g.seed_x) =
                static_cast<unsigned short>(g.label);
        }
    }

    return out;
}

std::vector<cv::Mat> build_final_grain_label_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth)
{
    std::vector<cv::Mat> out;
    out.reserve(depth);

    for (int z = 0; z < depth; ++z) {
        out.push_back(cv::Mat::zeros(height, width, CV_16UC1));
    }

    for (const FinalGrain& g : grains) {
        const unsigned short lab = static_cast<unsigned short>(g.label);

        for (long p : g.voxels) {
            int x = 0, y = 0, z = 0;
            decode_voxel(p, width, height, x, y, z);

            if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) {
                continue;
            }

            out[z].at<unsigned short>(y, x) = lab;
        }
    }

    return out;
}

void save_seed_csv(
    const std::string& csv_path,
    const std::vector<FinalGrain>& grains)
{
    std::ofstream csv(csv_path);
    csv
        << "label,node_id,score,volume,mean_intensity,fill_ratio,"
        << "dx,dy,dz,z_min,z_max,active_slices,longest_run,z_peak,area_peak,"
        << "track_score,cx,cy,cz,"
        << "seed_x,seed_y,seed_z,seed_value,seed_type,"
        << "alt,parent_alt,delta_parent\n";

    for (const FinalGrain& g : grains) {
        if (g.seed_x < 0 || g.seed_y < 0 || g.seed_z < 0) {
            continue;
        }

        csv
            << g.label << ","
            << g.node_id << ","
            << g.score << ","
            << g.geo.volume << ","
            << g.mean_intensity << ","
            << g.fill_ratio << ","
            << g.geo.dx << ","
            << g.geo.dy << ","
            << g.geo.dz << ","
            << g.track.z_min_active << ","
            << g.track.z_max_active << ","
            << g.track.active_slices << ","
            << g.track.longest_run << ","
            << g.track.z_peak << ","
            << g.track.area_peak << ","
            << g.track.track_score << ","
            << g.geo.cx << ","
            << g.geo.cy << ","
            << g.geo.cz << ","
            << g.seed_x << ","
            << g.seed_y << ","
            << g.seed_z << ","
            << g.seed_value << ","
            << (g.seed_from_dark_candidate ? "dark_ttos" : "support_fallback") << ","
            << g.alt << ","
            << g.parent_alt << ","
            << g.delta_parent
            << "\n";
    }
}
