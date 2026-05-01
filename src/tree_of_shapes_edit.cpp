#include "tree_of_shapes_edit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace
{
void refresh_ttos_attributes(Tree_of_shapes& tos)
{
    tos.enrich();
    tos.compute_area();
}

long interval_amplitude(const Node_tos* node)
{
    if (node == nullptr) return 0;
    return std::llabs(static_cast<long>(node->interval[1]) - static_cast<long>(node->interval[0]));
}

bool matches_leaf_filter(const Node_tos* node, const ToSLeafFilterParams& params)
{
    if (!is_active_leaf(node)) return false;
    if (node->area > params.max_area) return false;
    if (params.node_class_constraint != NA && node->node_class != params.node_class_constraint) return false;
    if (node->parent == nullptr) return false;

    const long delta_parent = std::llabs(node->alt - node->parent->alt);
    if (delta_parent > params.max_delta_parent) return false;
    if (interval_amplitude(node) > params.max_interval_amplitude) return false;

    return true;
}

long nearest_upper_bound(Node_tos* node)
{
    node->compute_boundaries();
    return node->upper_bound;
}

long nearest_lower_bound(Node_tos* node)
{
    node->compute_boundaries();
    return node->lower_bound;
}

struct QueueEntry
{
    unsigned int name = 0;
    Node_tos* node = nullptr;
};

struct QueueEntryGreater
{
    bool operator()(const QueueEntry& a, const QueueEntry& b) const
    {
        return a.name > b.name;
    }
};

void refill_candidate_heap(
    Tree_of_shapes& tos,
    const ToSLeafFilterParams& params,
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater>& heap,
    ToSPaperOperatorStats* stats,
    bool reset_initial_counter)
{
    while (!heap.empty()) heap.pop();

    if (reset_initial_counter && stats != nullptr) {
        stats->initial_candidate_leaves = 0;
    }

    for (auto& kv : tos.nodes) {
        Node_tos* node = kv.second.get();
        if (matches_leaf_filter(node, params)) {
            heap.push(QueueEntry{node->name, node});
            if (stats != nullptr) {
                stats->initial_candidate_leaves++;
            }
        }
    }
}

void print_single_line_progress(
    const std::string& label,
    std::size_t current,
    std::size_t total_hint,
    std::size_t active_hint)
{
    const std::size_t bar_width = 34;
    const double denom = static_cast<double>(std::max<std::size_t>(1, total_hint));
    const double ratio = std::min(1.0, static_cast<double>(current) / denom);
    const std::size_t filled =
        static_cast<std::size_t>(std::round(ratio * static_cast<double>(bar_width)));

    std::ostringstream oss;
    oss << "\r" << std::left << std::setw(14) << label << " [";
    for (std::size_t i = 0; i < bar_width; ++i) {
        oss << (i < filled ? '#' : '-');
    }
    oss << "] " << std::setw(7) << current
        << " it | active≈" << std::setw(6) << active_hint
        << " | " << std::fixed << std::setprecision(1) << (100.0 * ratio) << "%";
    std::cout << oss.str() << std::flush;
}
} // namespace

void process_tree_median(Tree_of_shapes& tos)
{
    std::queue<Node_tos*> queue;
    queue.push(tos.root);
    std::unordered_map<long, bool> visited;

    while (!queue.empty()) {
        Node_tos* node = queue.front();
        queue.pop();

        if (!node->root && !node->removed) {
            const long target_alt = node->get_median_neighbouring_value();
            node = tos.change_alt_of_node(node->name, target_alt);

            if (node != nullptr && !node->root && std::abs(node->parent->alt - target_alt) < 1) {
                tos.change_alt_of_node(node->name, node->parent->alt);
            }
        }

        for (Node_tos* child : node->children) {
            if (child != nullptr && !visited[child->name]) {
                queue.push(child);
                visited[child->name] = true;
            }
        }
    }

    refresh_ttos_attributes(tos);
}

void process_tree_proper_part(Tree_of_shapes& tos, long proper_part_value)
{
    std::queue<Node_tos*> queue;
    queue.push(tos.root);
    std::unordered_map<long, bool> visited;

    while (!queue.empty()) {
        Node_tos* node = queue.front();
        queue.pop();

        if (!node->root) {
            while (!node->root && !node->removed && node->area < proper_part_value) {
                node = tos.change_alt_of_node(node->name, node->parent->alt);
            }
        }

        for (Node_tos* child : node->children) {
            if (child != nullptr && !visited[child->name]) {
                queue.push(child);
                visited[child->name] = true;
            }
        }
    }

    refresh_ttos_attributes(tos);
}

void process_tree_proper_part_bottom_up(Tree_of_shapes& tos, long proper_part_value)
{
    process_tree_proper_part(tos, proper_part_value);
}

void process_tree_proper_part_bottom_up_consecutive(
    Tree_of_shapes& tos,
    long starting_value,
    long step,
    long end_value)
{
    const long safe_step = std::max<long>(1, step);
    for (long i = starting_value; i < end_value; i += safe_step) {
        process_tree_proper_part(tos, i);
    }
}

bool is_active_leaf(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed) return false;

    for (const Node_tos* child : node->children) {
        if (child != nullptr && !child->removed) {
            return false;
        }
    }
    return true;
}

void discard_node_paper(Tree_of_shapes&, Node_tos* node, ToSPaperOperatorStats* stats)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) return;

    node->fuse_to_parent();
    if (stats != nullptr) stats->discarded_nodes++;
}

Node_tos* short_range_shift_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    long target_alt,
    ToSPaperOperatorStats* stats)
{
    if (node == nullptr || node->removed) return node;

    node->compute_boundaries();
    const long parent_alt = (node->parent != nullptr) ? node->parent->alt : node->alt;
    const long bounded_alt = node->bound_value(target_alt);

    if (node->is_strictly_between_bounds(bounded_alt)) {
        node->alt = bounded_alt;
        if (stats != nullptr) stats->shifted_nodes++;
        return node;
    }

    std::vector<Node_tos*> impacted_children;
    if (node->lower_bound != -1 && bounded_alt == node->lower_bound) {
        impacted_children = node->get_lower_bound_children();
    } else if (node->upper_bound != -1 && bounded_alt == node->upper_bound) {
        impacted_children = node->get_upper_bound_children();
    }

    std::vector<Node_tos*> children_to_discard;
    children_to_discard.reserve(impacted_children.size());

    for (Node_tos* child : impacted_children) {
        if (child != nullptr && !child->removed && child->parent == node) {
            children_to_discard.push_back(child);
        }
    }

    for (Node_tos* child : children_to_discard) {
        discard_node_paper(tos, child, stats);
    }

    if (node->removed) return node->parent;

    node->alt = bounded_alt;
    if (stats != nullptr) stats->shifted_nodes++;

    if (node->parent != nullptr && bounded_alt == parent_alt) {
        Node_tos* parent = node->parent;
        discard_node_paper(tos, node, stats);
        return parent;
    }

    return node;
}

Node_tos* long_range_shift_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    long target_alt,
    ToSPaperOperatorStats* stats)
{
    if (node == nullptr || node->removed) return node;

    while (node != nullptr && !node->removed && node->alt != target_alt) {
        const long upper = nearest_upper_bound(node);
        const long lower = nearest_lower_bound(node);

        long intermediate_target = target_alt;
        if (target_alt > node->alt && upper != -1 && target_alt > upper) {
            intermediate_target = upper;
        } else if (target_alt < node->alt && lower != -1 && target_alt < lower) {
            intermediate_target = lower;
        }

        if (intermediate_target == node->alt) break;
        node = short_range_shift_paper(tos, node, intermediate_target, stats);
    }

    return node;
}

ToSPaperOperatorStats apply_grain_filter_paper(
    Tree_of_shapes& tos,
    long lambda_leaf,
    bool show_progress,
    std::size_t progress_every,
    std::size_t refresh_every)
{
    ToSLeafFilterParams params;
    params.max_area = lambda_leaf;
    params.progress_label = "ToSConOp";
    return apply_selective_leaf_filter(tos, params, show_progress, progress_every, refresh_every);
}

ToSPaperOperatorStats apply_selective_leaf_filter(
    Tree_of_shapes& tos,
    const ToSLeafFilterParams& params,
    bool show_progress,
    std::size_t progress_every,
    std::size_t refresh_every)
{
    ToSPaperOperatorStats stats;
    refresh_ttos_attributes(tos);

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> heap;
    refill_candidate_heap(tos, params, heap, &stats, true);

    const std::string label =
        params.progress_label.empty() ? "ToSConOp" : params.progress_label;

    while (!heap.empty()) {
        QueueEntry entry = heap.top();
        heap.pop();

        Node_tos* chosen = entry.node;
        if (!matches_leaf_filter(chosen, params)) continue;
        if (chosen->parent == nullptr) continue;

        stats.iterations++;

        Node_tos* old_parent = chosen->parent;
        const long target_alt = old_parent->alt;
        Node_tos* result = long_range_shift_paper(tos, chosen, target_alt, &stats);

        if (refresh_every > 0 && stats.iterations % refresh_every == 0) {
            refresh_ttos_attributes(tos);
            refill_candidate_heap(tos, params, heap, nullptr, false);
        } else {
            if (result != nullptr && matches_leaf_filter(result, params)) {
                heap.push(QueueEntry{result->name, result});
            }
            if (old_parent != nullptr && matches_leaf_filter(old_parent, params)) {
                heap.push(QueueEntry{old_parent->name, old_parent});
            }
            if (result != nullptr && result->parent != nullptr && matches_leaf_filter(result->parent, params)) {
                heap.push(QueueEntry{result->parent->name, result->parent});
            }
        }

        if (show_progress && progress_every > 0 && (stats.iterations % progress_every == 0)) {
            print_single_line_progress(label, stats.iterations, stats.initial_candidate_leaves, heap.size());
        }
    }

    refresh_ttos_attributes(tos);

    if (show_progress) {
        print_single_line_progress(label, stats.iterations, std::max<std::size_t>(1, stats.iterations), 0);
        std::cout << std::endl;
    }

    return stats;
}

void flatten_tree_for_support(Tree_of_shapes& tos, long lambda_support)
{
    ToSLeafFilterParams params;
    params.max_area = std::max<long>(1, lambda_support);
    params.node_class_constraint = MAX_TREE;
    params.progress_label = "LegacyBright";
    apply_selective_leaf_filter(tos, params, false, 0, 256);
}

void flatten_tree_for_seed(Tree_of_shapes& tos, long lambda_seed)
{
    ToSLeafFilterParams params;
    params.max_area = std::max<long>(1, lambda_seed);
    params.node_class_constraint = MIN_TREE;
    params.progress_label = "LegacyDark";
    apply_selective_leaf_filter(tos, params, false, 0, 256);
}
