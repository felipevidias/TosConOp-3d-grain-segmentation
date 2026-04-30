#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "node_tos.h"
#include "ttos_grain_types.h"

double clamp01(double v);
double distance3d(
    double x1, double y1, double z1,
    double x2, double y2, double z2);

bool is_ancestor(Node_tos* possible_ancestor, Node_tos* node);

void decode_voxel(long p, int width, int height, int& x, int& y, int& z);

double percentile_from_volume(const std::vector<cv::Mat>& volume, double p);

void collect_subtree_voxels(const Node_tos* node, std::vector<long>& voxels);

GeoStats compute_geometry(
    const std::vector<long>& voxels,
    int width,
    int height,
    int depth);

double compute_mean_intensity(
    const std::vector<long>& voxels,
    const std::vector<cv::Mat>& volume,
    int width,
    int height,
    int depth);

double bbox_overlap_ratio_min(const GeoStats& a, const GeoStats& b);

ZTrackStats compute_ztrack(
    const std::vector<long>& voxels,
    int width,
    int height,
    int depth);

LocalStats compute_local_stats(
    const std::vector<cv::Mat>& volume,
    int cx,
    int cy,
    int cz);

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
    int& out_value);

bool choose_fallback_seed_from_support(
    const std::vector<long>& voxels,
    const GeoStats& geo,
    int width,
    int height,
    int depth,
    int& sx,
    int& sy,
    int& sz);

void trim_support_candidates(
    std::vector<SupportCandidate>& candidates,
    std::size_t max_keep);
