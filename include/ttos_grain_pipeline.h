#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tree_of_shapes.h"
#include "ttos_grain_params.h"
#include "ttos_grain_types.h"

std::vector<SupportCandidate> extract_support_candidates(
    Tree_of_shapes& tos_support,
    const std::vector<cv::Mat>& volume,
    const TtosGrainParams& params,
    double p30,
    double p60);

std::vector<SupportCandidate> nms_support_candidates(
    const std::vector<SupportCandidate>& candidates,
    const TtosGrainParams& params);

std::vector<FinalGrain> make_final_grains_from_supports(
    const std::vector<SupportCandidate>& supports,
    Tree_of_shapes& tos_support);

std::vector<cv::Mat> build_support_label_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth);

void assign_dark_ttos_seeds(
    Tree_of_shapes& tos_seed,
    const std::vector<cv::Mat>& volume,
    const std::vector<cv::Mat>& support_labels,
    const TtosGrainParams& params,
    double p50,
    std::vector<FinalGrain>& grains);

void assign_support_fallback_seeds(
    const std::vector<cv::Mat>& volume,
    std::vector<FinalGrain>& grains);

std::vector<cv::Mat> build_marker_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth);

std::vector<cv::Mat> build_final_grain_label_volume(
    const std::vector<FinalGrain>& grains,
    int width,
    int height,
    int depth);

void save_seed_csv(
    const std::string& csv_path,
    const std::vector<FinalGrain>& grains);
