#include "ttos_grain_pipeline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stack>
#include <utility>
#include <vector>

#include "ttos_grain_analysis.h"

namespace
{
bool is_support_node(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) return false;
    return node->node_class == MAX_TREE && node->alt > node->parent->alt;
}

bool is_seed_node(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) return false;
    return node->node_class == MIN_TREE && node->alt < node->parent->alt;
}

bool same_tree_branch(Node_tos* a, Node_tos* b)
{
    return is_ancestor(a, b) || is_ancestor(b, a);
}

double node_interval_amplitude(const Node_tos* node)
{
    if (node == nullptr) return 0.0;
    return static_cast<double>(std::abs(node->interval[1] - node->interval[0]));
}

double normalized_log_area(long area, long min_area, long max_area)
{
    const double a = std::log(static_cast<double>(std::max<long>(1, area)) + 1.0);
    const double a0 = std::log(static_cast<double>(std::max<long>(1, min_area)) + 1.0);
    const double a1 = std::log(static_cast<double>(std::max<long>(min_area + 1, max_area)) + 1.0);
    return clamp01((a - a0) / std::max(1e-9, a1 - a0));
}

double branch_persistence(Node_tos* node)
{
    if (node == nullptr || node->parent == nullptr) return 0.0;

    Node_tos* cur = node;
    double best = 0.0;
    int steps = 0;

    while (cur != nullptr && cur->parent != nullptr && steps < 14) {
        Node_tos* p = cur->parent;
        if (p->node_class != node->node_class) break;

        const double delta = clamp01(static_cast<double>(std::llabs(cur->alt - p->alt)) / 90.0);
        best = std::max(best, delta);
        cur = p;
        steps++;
    }

    return best;
}

struct SeedEvidence
{
    int count_strong = 0;
    double best_score = 0.0;
    long best_node_name = -1;
};

SeedEvidence evaluate_seed_descendants(
    Node_tos* support_node,
    const TtosGrainParams& params)
{
    SeedEvidence ev;
    if (support_node == nullptr) return ev;

    std::stack<std::pair<Node_tos*, int>> st;
    st.push({support_node, 0});

    while (!st.empty()) {
        auto [node, depth] = st.top();
        st.pop();

        if (depth > params.max_seed_descendant_depth) continue;

        if (depth > 0 && is_seed_node(node)) {
            if (node->area >= params.seed_area_min && node->area <= params.seed_area_max) {
                const long delta_parent = std::llabs(node->alt - node->parent->alt);
                if (delta_parent >= params.seed_min_delta_parent) {
                    const double delta_score = clamp01(static_cast<double>(delta_parent) / 70.0);
                    const double interval_score = clamp01(node_interval_amplitude(node) / 70.0);
                    const double area_score =
                        1.0 - std::abs(0.45 - normalized_log_area(node->area, params.seed_area_min, params.seed_area_max));
                    const double depth_score = clamp01(
                        1.0 - static_cast<double>(depth) / std::max(1, params.max_seed_descendant_depth));

                    const double tree_seed_score =
                        0.38 * delta_score +
                        0.28 * interval_score +
                        0.18 * area_score +
                        0.16 * depth_score;

                    if (tree_seed_score > 0.018) {
                        ev.count_strong++;
                    }
                    if (tree_seed_score > ev.best_score) {
                        ev.best_score = tree_seed_score;
                        ev.best_node_name = static_cast<long>(node->name);
                    }
                }
            }
        }

        for (Node_tos* child : node->children) {
            if (child != nullptr && !child->removed) {
                st.push({child, depth + 1});
            }
        }
    }

    return ev;
}

double support_tree_score(
    Node_tos* node,
    const TtosGrainParams& params,
    const SeedEvidence& seed_ev)
{
    if (node == nullptr || node->parent == nullptr) return 0.0;

    const double delta_score =
        clamp01(static_cast<double>(std::llabs(node->alt - node->parent->alt)) / 70.0);
    const double interval_score =
        clamp01(node_interval_amplitude(node) / 70.0);
    const double area_score =
        normalized_log_area(node->area, params.support_area_min, params.support_area_max);
    const double persistence_score = branch_persistence(node);
    const double seed_score = clamp01(seed_ev.best_score / 0.60);

    // Very soft ambiguity penalty: dense regions were being over-suppressed.
    const double ambiguity_penalty =
        (seed_ev.count_strong > 30) ? std::min(0.03, 0.0025 * static_cast<double>(seed_ev.count_strong - 30)) : 0.0;

    return
        0.22 * delta_score +
        0.20 * interval_score +
        0.18 * area_score +
        0.22 * persistence_score +
        0.18 * seed_score -
        ambiguity_penalty;
}

SupportCandidate build_support_candidate_from_node(
    Node_tos* node,
    Tree_of_shapes& tos_filtered,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams& params)
{
    SupportCandidate c;
    c.node = node;
    c.area = node->area;
    c.alt = node->alt;
    c.parent_alt = node->parent != nullptr ? node->parent->alt : node->alt;
    c.delta_parent = std::abs(static_cast<int>(c.parent_alt - c.alt));

    std::vector<long> voxels;
    collect_subtree_voxels(node, voxels);

    c.geo = compute_geometry(voxels, tos_filtered.width, tos_filtered.height, tos_filtered.depth);
    c.track = compute_ztrack(voxels, tos_filtered.width, tos_filtered.height, tos_filtered.depth);
    c.mean_intensity = compute_mean_intensity(voxels, volume, tos_filtered.width, tos_filtered.height, tos_filtered.depth);

    const double bbox_volume =
        static_cast<double>(std::max(1, c.geo.dx)) * std::max(1, c.geo.dy) * std::max(1, c.geo.dz);
    c.fill_ratio = bbox_volume > 0.0 ? static_cast<double>(c.geo.volume) / bbox_volume : 0.0;

    const SeedEvidence seed_ev = evaluate_seed_descendants(node, params);
    c.strong_seed_descendants = seed_ev.count_strong;
    c.seed_descendant_score = seed_ev.best_score;
    c.dominant_seed_node_id = seed_ev.best_node_name;
    c.tree_support_score = support_tree_score(node, params, seed_ev);
    c.score = c.tree_support_score + 0.20 * c.seed_descendant_score;
    return c;
}

double area_ratio_min(long a, long b)
{
    const double aa = static_cast<double>(std::max<long>(1, a));
    const double bb = static_cast<double>(std::max<long>(1, b));
    return std::min(aa, bb) / std::max(aa, bb);
}

bool current_should_replace_accepted(const SupportCandidate& current, const SupportCandidate& accepted)
{
    if (!is_ancestor(accepted.node, current.node)) return false;

    const bool smaller =
        current.area <= static_cast<long>(0.90 * std::max<long>(1, accepted.area));
    const bool similar_seed =
        current.seed_descendant_score >= 0.82 * accepted.seed_descendant_score;
    const bool similar_support =
        current.tree_support_score + 0.005 >= accepted.tree_support_score;
    const bool not_more_ambiguous =
        current.strong_seed_descendants <= accepted.strong_seed_descendants + 3;

    return smaller && similar_seed && similar_support && not_more_ambiguous;
}

bool near_duplicate_same_branch(const SupportCandidate& a, const SupportCandidate& b)
{
    const double ar = area_ratio_min(a.area, b.area);
    const bool similar_seed = std::abs(a.seed_descendant_score - b.seed_descendant_score) < 0.015;
    return ar > 0.55 && similar_seed;
}
} // namespace

std::vector<SupportCandidate> extract_support_candidates(
    Tree_of_shapes& tos_filtered,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams& params,
    double,
    double)
{
    std::vector<SupportCandidate> candidates;
    candidates.reserve(24000);

    tos_filtered.enrich();
    tos_filtered.compute_area();

    for (auto const& item : tos_filtered.nodes) {
        Node_tos* node = item.second.get();

        if (!is_support_node(node)) continue;
        if (node->area < params.support_area_min || node->area > params.support_area_max) continue;

        SupportCandidate c = build_support_candidate_from_node(node, tos_filtered, volume, params);

        if (c.tree_support_score < params.support_tree_score_min) continue;
        if (c.seed_descendant_score < params.support_seed_descendant_score_min) continue;
        if (c.dominant_seed_node_id < 0) continue;

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
            if (a.seed_descendant_score != b.seed_descendant_score) return a.seed_descendant_score > b.seed_descendant_score;
            if (a.area != b.area) return a.area < b.area;
            if (a.tree_support_score != b.tree_support_score) return a.tree_support_score > b.tree_support_score;
            return a.strong_seed_descendants < b.strong_seed_descendants;
        });

    return candidates;
}

std::vector<SupportCandidate> nms_support_candidates(
    const std::vector<SupportCandidate>& candidates,
    const TtosGrainParams& params)
{
    std::vector<SupportCandidate> accepted;
    accepted.reserve(candidates.size());

    for (const SupportCandidate& cand : candidates) {
        if (accepted.size() >= params.max_final_grains) break;

        bool reject = false;

        for (std::size_t i = 0; i < accepted.size(); ++i) {
            const SupportCandidate& acc = accepted[i];

            if (cand.dominant_seed_node_id == acc.dominant_seed_node_id) {
                if (current_should_replace_accepted(cand, acc)) {
                    accepted[i] = cand;
                }
                reject = true;
                break;
            }

            if (same_tree_branch(cand.node, acc.node)) {
                if (near_duplicate_same_branch(cand, acc) || current_should_replace_accepted(cand, acc)) {
                    if (current_should_replace_accepted(cand, acc)) {
                        accepted[i] = cand;
                    }
                    reject = true;
                    break;
                }
                // Same branch but plausibly distinct grain: keep both.
            }
        }

        if (!reject) {
            accepted.push_back(cand);
        }
    }

    return accepted;
}

std::vector<FinalGrain> make_final_grains_from_supports(
    const std::vector<SupportCandidate>& supports,
    Tree_of_shapes& tos_filtered,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams&)
{
    std::vector<FinalGrain> grains;
    grains.reserve(supports.size());

    int label = 1;
    for (const SupportCandidate& s : supports) {
        if (s.node == nullptr) continue;

        std::vector<long> voxels;
        collect_subtree_voxels(s.node, voxels);

        FinalGrain g;
        g.label = label++;
        g.node_id = s.node->name;
        g.voxels = std::move(voxels);
        g.geo = compute_geometry(g.voxels, tos_filtered.width, tos_filtered.height, tos_filtered.depth);
        g.track = compute_ztrack(g.voxels, tos_filtered.width, tos_filtered.height, tos_filtered.depth);
        g.area = static_cast<long>(g.voxels.size());
        g.alt = s.node->alt;
        g.parent_alt = s.node->parent != nullptr ? s.node->parent->alt : s.node->alt;
        g.delta_parent = std::abs(static_cast<int>(g.parent_alt - g.alt));
        g.mean_intensity = compute_mean_intensity(g.voxels, volume, tos_filtered.width, tos_filtered.height, tos_filtered.depth);

        const double bbox_volume =
            static_cast<double>(std::max(1, g.geo.dx)) * std::max(1, g.geo.dy) * std::max(1, g.geo.dz);
        g.fill_ratio = bbox_volume > 0.0 ? static_cast<double>(g.geo.volume) / bbox_volume : 0.0;

        g.score = s.score;
        g.support_promoted_steps = 0;
        g.support_tree_score = s.tree_support_score;
        g.support_seed_descendant_score = s.seed_descendant_score;
        g.support_strong_seed_descendants = s.strong_seed_descendants;

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
            if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) continue;
            if (out[z].at<unsigned short>(y, x) == 0) {
                out[z].at<unsigned short>(y, x) = lab;
            }
        }
    }

    return out;
}

void assign_dark_ttos_seeds(
    Tree_of_shapes& tos_filtered,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams& params,
    double,
    std::vector<FinalGrain>& grains)
{
    tos_filtered.enrich();
    tos_filtered.compute_area();

    for (FinalGrain& g : grains) {
        auto it = tos_filtered.nodes.find(g.node_id);
        if (it == tos_filtered.nodes.end()) continue;

        Node_tos* support_node = it->second.get();

        std::stack<std::pair<Node_tos*, int>> st;
        st.push({support_node, 0});

        double best_score = -1e100;

        while (!st.empty()) {
            auto [node, depth] = st.top();
            st.pop();

            if (depth > params.max_seed_descendant_depth) continue;

            if (depth > 0 && is_seed_node(node)) {
                if (node->area >= params.seed_area_min && node->area <= params.seed_area_max) {
                    const int delta = std::abs(static_cast<int>(node->parent->alt - node->alt));
                    if (delta >= params.seed_min_delta_parent) {
                        const double delta_score = clamp01(static_cast<double>(delta) / 70.0);
                        const double interval_score = clamp01(node_interval_amplitude(node) / 70.0);
                        const double area_score =
                            1.0 - std::abs(0.45 - normalized_log_area(node->area, params.seed_area_min, params.seed_area_max));
                        const double depth_score =
                            clamp01(1.0 - static_cast<double>(depth) / std::max(1, params.max_seed_descendant_depth));

                        const double tree_score =
                            0.38 * delta_score +
                            0.28 * interval_score +
                            0.18 * area_score +
                            0.16 * depth_score;

                        if (tree_score > best_score) {
                            std::vector<long> voxels;
                            collect_subtree_voxels(node, voxels);

                            GeoStats geo = compute_geometry(voxels, tos_filtered.width, tos_filtered.height, tos_filtered.depth);
                            int sx = 0, sy = 0, sz = 0, seed_value = 255;
                            if (find_representative_voxel_of_node(
                                    voxels, volume, geo, tos_filtered.width, tos_filtered.height, tos_filtered.depth,
                                    sx, sy, sz, seed_value)) {
                                best_score = tree_score;
                                g.seed_x = sx;
                                g.seed_y = sy;
                                g.seed_z = sz;
                                g.seed_value = seed_value;
                                g.seed_from_dark_candidate = true;
                                g.seed_found_in_descendants = true;
                                g.seed_tree_score = tree_score;
                            }
                        }
                    }
                }
            }

            for (Node_tos* child : node->children) {
                if (child != nullptr && !child->removed) {
                    st.push({child, depth + 1});
                }
            }
        }
    }

    if (params.allow_support_fallback_seed) {
        assign_support_fallback_seeds(volume, grains);
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
        if (g.seed_x >= 0 && g.seed_y >= 0 && g.seed_z >= 0) continue;

        int sx = 0, sy = 0, sz = 0;
        if (!choose_fallback_seed_from_support(g.voxels, g.geo, width, height, depth, sx, sy, sz)) {
            continue;
        }

        g.seed_x = sx;
        g.seed_y = sy;
        g.seed_z = sz;
        g.seed_value = volume[sz].at<uchar>(sy, sx);
        g.seed_from_dark_candidate = false;
        g.seed_found_in_descendants = false;
        g.seed_tree_score = 0.0;
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
        if (g.seed_x < 0 || g.seed_y < 0 || g.seed_z < 0) continue;
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
            if (z < 0 || z >= depth || y < 0 || y >= height || x < 0 || x >= width) continue;
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
    csv << "label,node_id,score,volume,mean_intensity,fill_ratio,"
           "support_promoted_steps,support_tree_score,support_seed_descendant_score,"
           "support_strong_seed_descendants,"
           "dx,dy,dz,z_min,z_max,active_slices,longest_run,z_peak,area_peak,track_score,"
           "cx,cy,cz,seed_x,seed_y,seed_z,seed_value,seed_type,seed_tree_score,"
           "alt,parent_alt,delta_parent\n";

    for (const FinalGrain& g : grains) {
        csv << g.label << ","
            << g.node_id << ","
            << g.score << ","
            << g.geo.volume << ","
            << g.mean_intensity << ","
            << g.fill_ratio << ","
            << g.support_promoted_steps << ","
            << g.support_tree_score << ","
            << g.support_seed_descendant_score << ","
            << g.support_strong_seed_descendants << ","
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
            << (g.seed_from_dark_candidate ? "dark_descendant" : "support_fallback") << ","
            << g.seed_tree_score << ","
            << g.alt << ","
            << g.parent_alt << ","
            << g.delta_parent
            << "\n";
    }
}
