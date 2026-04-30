#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "ttos_grain_types.h"

cv::Vec3b color_from_label(int label);

void paint_seed_qc(
    std::vector<cv::Mat>& qc,
    int x,
    int y,
    int z,
    int label,
    const cv::Vec3b& color);

void build_persistent_track_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains,
    std::vector<cv::Mat>& out_rgb);

cv::Mat render_seed_orthoview_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains);

cv::Mat render_track_orthoview_qc(
    const std::vector<cv::Mat>& volume,
    const std::vector<FinalGrain>& grains);
