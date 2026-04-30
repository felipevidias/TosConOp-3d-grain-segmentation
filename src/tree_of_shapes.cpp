#include "tree_of_shapes.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

Tree_of_shapes::Tree_of_shapes(const std::vector<cv::Mat>& slices)
{
    if (slices.empty()) {
        throw std::runtime_error("Empty volume passed to Tree_of_shapes.");
    }

    depth = static_cast<int>(slices.size());
    height = slices[0].rows;
    width = slices[0].cols;
    img_size = static_cast<long>(depth) * height * width;

    volume_data.resize(img_size);

    for (int z = 0; z < depth; ++z) {
        if (slices[z].rows != height || slices[z].cols != width || slices[z].type() != CV_8U) {
            throw std::runtime_error("All slices must have the same shape and CV_8U type.");
        }

        std::memcpy(
            volume_data.data() + static_cast<long>(z) * height * width,
            slices[z].data,
            static_cast<size_t>(height) * width);
    }

    auto img_xt = xt::adapt(
        volume_data.data(),
        img_size,
        xt::no_ownership(),
        std::vector<size_t>{static_cast<size_t>(depth),
                            static_cast<size_t>(height),
                            static_cast<size_t>(width)});

    tos = hg::component_tree_tree_of_shapes_image(img_xt);
    tos.tree.compute_children();
    parents = tos.tree.parents();

    pre_process_tos();
    enrich();
    compute_area();
}

void Tree_of_shapes::pre_process_tos()
{
    nodes.clear();

    const long id_root = static_cast<long>(tos.tree.root()) - img_size;
    const int alt_root = tos.altitudes[tos.tree.root()];

    nodes[id_root] = std::make_unique<Node_tos>(id_root, alt_root, nullptr);
    root = nodes[id_root].get();
    root->root = true;
    highest_value = root->alt;

    std::vector<long> stack = {id_root};

    while (!stack.empty()) {
        const long current = stack.back();
        stack.pop_back();

        Node_tos* current_node = nodes[current].get();

        for (auto child : tos.tree.children(current + img_size)) {
            if (child < img_size) {
                current_node->proper_part.emplace_back(child);
                continue;
            }

            const long id_child = static_cast<long>(child) - img_size;
            const int child_alt = tos.altitudes[child];
            nodes[id_child] = std::make_unique<Node_tos>(id_child, child_alt, current_node);
            stack.push_back(id_child);
        }
    }
}

void Tree_of_shapes::enrich()
{
    if (root == nullptr) {
        enriched = false;
        return;
    }

    std::queue<Node_tos*> q;
    q.push(root);

    while (!q.empty()) {
        Node_tos* node = q.front();
        q.pop();

        node->enrich(highest_value);

        for (Node_tos* child : node->children) {
            if (child != nullptr && !child->removed) {
                q.push(child);
            }
        }
    }

    enriched = true;
}

void Tree_of_shapes::compute_area()
{
    for (auto& kv : nodes) {
        kv.second->area = 0;
    }

    if (root != nullptr) {
        root->compute_area();
    }
}

Node_tos* Tree_of_shapes::change_alt_of_node(unsigned int node_name, long new_alt)
{
    auto it = nodes.find(node_name);
    if (it == nodes.end()) {
        return nullptr;
    }

    Node_tos* node = it->second.get();
    while (node != nullptr && node->alt != new_alt) {
        node = node->change_node_altitude_in_bounds(new_alt);
    }
    return node;
}

std::vector<cv::Mat> Tree_of_shapes::reconstruct_image_3d()
{
    std::vector<cv::Mat> output_slices;
    output_slices.reserve(depth);

    for (int z = 0; z < depth; ++z) {
        output_slices.push_back(cv::Mat::zeros(height, width, CV_8UC1));
    }

    if (root == nullptr) {
        return output_slices;
    }

    std::queue<const Node_tos*> q;
    q.push(root);

    const long slice_area = static_cast<long>(height) * width;

    while (!q.empty()) {
        const Node_tos* node = q.front();
        q.pop();

        for (long px : node->proper_part) {
            const int z = static_cast<int>(px / slice_area);
            const long rem = px % slice_area;
            const int y = static_cast<int>(rem / width);
            const int x = static_cast<int>(rem % width);

            if (z >= 0 && z < depth && y >= 0 && y < height && x >= 0 && x < width) {
                output_slices[z].at<uchar>(y, x) =
                    static_cast<uchar>(std::clamp<long>(node->alt, 0, 255));
            }
        }

        for (const Node_tos* child : node->children) {
            if (child != nullptr && !child->removed) {
                q.push(child);
            }
        }
    }

    return output_slices;
}

cv::Mat Tree_of_shapes::reconstruct_node_colored_image()
{
    return cv::Mat();
}

long Tree_of_shapes::nb_nodes()
{
    return static_cast<long>(nodes.size());
}
