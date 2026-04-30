#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <queue>

#include <higra/image/tree_of_shapes.hpp>
#include "node_tos.h"
#include <opencv2/opencv.hpp>
#include <xtensor/xadapt.hpp>

struct Tree_of_shapes
{
    Tree_of_shapes(const std::vector<cv::Mat> &slices);
    void pre_process_tos();

    void enrich();
    void compute_area();

    Node_tos * change_alt_of_node(unsigned int node_name, long new_alt);

    std::vector<cv::Mat> reconstruct_image_3d();
    cv::Mat reconstruct_node_colored_image();

    long nb_nodes();

    std::vector<uchar> volume_data;
    int width, height, depth;
    long img_size;
    int highest_value;

    hg::node_weighted_tree<hg::tree, hg::array_1d<uchar>> tos;
    hg::array_1d<hg::index_t> parents;

    Node_tos *root = nullptr;
    std::unordered_map<long, std::unique_ptr<Node_tos>> nodes;

    bool enriched = false;
};
