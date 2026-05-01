#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include "tree_of_shapes.h"

struct ToSPaperOperatorStats
{
    std::size_t iterations = 0;
    std::size_t discarded_nodes = 0;
    std::size_t shifted_nodes = 0;
    std::size_t initial_candidate_leaves = 0;
};

struct ToSLeafFilterParams
{
    long max_area = 15;
    long max_delta_parent = std::numeric_limits<long>::max();
    long max_interval_amplitude = std::numeric_limits<long>::max();
    Node_class node_class_constraint = NA;
    std::string progress_label = "";
};

bool is_active_leaf(const Node_tos* node);

void discard_node_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    ToSPaperOperatorStats* stats = nullptr);

Node_tos* short_range_shift_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    long target_alt,
    ToSPaperOperatorStats* stats = nullptr);

Node_tos* long_range_shift_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    long target_alt,
    ToSPaperOperatorStats* stats = nullptr);

ToSPaperOperatorStats apply_grain_filter_paper(
    Tree_of_shapes& tos,
    long lambda_leaf,
    bool show_progress = false,
    std::size_t progress_every = 100,
    std::size_t refresh_every = 256);

ToSPaperOperatorStats apply_selective_leaf_filter(
    Tree_of_shapes& tos,
    const ToSLeafFilterParams& params,
    bool show_progress = false,
    std::size_t progress_every = 100,
    std::size_t refresh_every = 256);

void flatten_tree_for_support(Tree_of_shapes& tos, long lambda_support);
void flatten_tree_for_seed(Tree_of_shapes& tos, long lambda_seed);

// Legacy helpers kept for compatibility.
void process_tree_median(Tree_of_shapes& tos);
void process_tree_proper_part(Tree_of_shapes& tos, long proper_part_value);
void process_tree_proper_part_bottom_up(Tree_of_shapes& tos, long proper_part_value);
void process_tree_proper_part_bottom_up_consecutive(
    Tree_of_shapes& tos,
    long starting_value,
    long step,
    long end_value);
