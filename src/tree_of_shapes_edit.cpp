#include "tree_of_shapes_edit.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

namespace
{
void refresh_ttos_attributes(Tree_of_shapes& tos)
{
    tos.enrich();
    tos.compute_area();
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

bool is_candidate_leaf(const Node_tos* node, long lambda_leaf)
{
    return is_active_leaf(node) && node->area <= lambda_leaf;
}
}

// To each node is assigned the median value of its neighborhood (parent + children).
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
    if (tos.root == nullptr) return;

    std::queue<Node_tos*> queue;
    std::vector<Node_tos*> order;
    std::unordered_map<long, bool> visited;

    queue.push(tos.root);

    while (!queue.empty()) {
        Node_tos* node = queue.front();
        queue.pop();

        order.push_back(node);

        for (Node_tos* child : node->children) {
            if (child != nullptr && !visited[child->name]) {
                queue.push(child);
                visited[child->name] = true;
            }
        }
    }

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node_tos* node = *it;

        if (!node->root && !node->removed) {
            while (!node->root && !node->removed && node->area < proper_part_value) {
                node = tos.change_alt_of_node(node->name, node->parent->alt);
            }
        }
    }

    refresh_ttos_attributes(tos);
}

void process_tree_proper_part_bottom_up_consecutive(
    Tree_of_shapes& tos,
    long starting_value,
    long step,
    long end_value)
{
    const long safe_step = std::max<long>(1, step);

    for (long i = starting_value; i < end_value; i += safe_step) {
        process_tree_proper_part_bottom_up(tos, i);
    }
}

bool is_active_leaf(const Node_tos* node)
{
    if (node == nullptr || node->root || node->removed) {
        return false;
    }

    for (const Node_tos* child : node->children) {
        if (child != nullptr && !child->removed) {
            return false;
        }
    }

    return true;
}

void discard_node_paper(
    Tree_of_shapes&,
    Node_tos* node,
    ToSPaperOperatorStats* stats)
{
    if (node == nullptr || node->root || node->removed || node->parent == nullptr) {
        return;
    }

    node->fuse_to_parent();

    if (stats != nullptr) {
        stats->discarded_nodes++;
    }
}

Node_tos* short_range_shift_paper(
    Tree_of_shapes& tos,
    Node_tos* node,
    long target_alt,
    ToSPaperOperatorStats* stats)
{
    if (node == nullptr || node->removed) {
        return node;
    }

    node->compute_boundaries();

    const long parent_alt = (node->parent != nullptr) ? node->parent->alt : node->alt;
    const long bounded_alt = node->bound_value(target_alt);

    if (node->is_strictly_between_bounds(bounded_alt)) {
        node->alt = bounded_alt;
        if (stats != nullptr) {
            stats->shifted_nodes++;
        }
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

    if (node->removed) {
        return node->parent;
    }

    node->alt = bounded_alt;
    if (stats != nullptr) {
        stats->shifted_nodes++;
    }

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
    if (node == nullptr || node->removed) {
        return node;
    }

    while (node != nullptr && !node->removed && node->alt != target_alt) {
        const long upper = nearest_upper_bound(node);
        const long lower = nearest_lower_bound(node);

        long intermediate_target = target_alt;

        if (target_alt > node->alt && upper != -1 && target_alt > upper) {
            intermediate_target = upper;
        } else if (target_alt < node->alt && lower != -1 && target_alt < lower) {
            intermediate_target = lower;
        }

        if (intermediate_target == node->alt) {
            break;
        }

        node = short_range_shift_paper(tos, node, intermediate_target, stats);
    }

    return node;
}

ToSPaperOperatorStats apply_grain_filter_paper(
    Tree_of_shapes& tos,
    long lambda_leaf,
    bool verbose,
    std::size_t progress_every)
{
    ToSPaperOperatorStats stats;
    refresh_ttos_attributes(tos);

    struct Entry {
        unsigned int name = 0;
        Node_tos* node = nullptr;
    };

    struct EntryGreater {
        bool operator()(const Entry& a, const Entry& b) const
        {
            return a.name > b.name;
        }
    };

    std::priority_queue<Entry, std::vector<Entry>, EntryGreater> heap;

    for (auto& kv : tos.nodes) {
        Node_tos* node = kv.second.get();
        if (is_candidate_leaf(node, lambda_leaf)) {
            heap.push(Entry{node->name, node});
            stats.initial_candidate_leaves++;
        }
    }

    while (!heap.empty()) {
        Entry entry = heap.top();
        heap.pop();

        Node_tos* chosen = entry.node;

        if (!is_candidate_leaf(chosen, lambda_leaf)) {
            continue;
        }

        if (chosen->parent == nullptr) {
            continue;
        }

        stats.iterations++;

        Node_tos* old_parent = chosen->parent;
        const long target_alt = old_parent->alt;

        Node_tos* result = long_range_shift_paper(tos, chosen, target_alt, &stats);

        if (result != nullptr && is_candidate_leaf(result, lambda_leaf)) {
            heap.push(Entry{result->name, result});
        }

        if (old_parent != nullptr && is_candidate_leaf(old_parent, lambda_leaf)) {
            heap.push(Entry{old_parent->name, old_parent});
        }

        if (result != nullptr && result->parent != nullptr && is_candidate_leaf(result->parent, lambda_leaf)) {
            heap.push(Entry{result->parent->name, result->parent});
        }

        if (verbose && progress_every > 0 && (stats.iterations % progress_every == 0)) {
            std::cout << "  operator progress: iteration " << stats.iterations
                      << " | shifted=" << stats.shifted_nodes
                      << " | discarded=" << stats.discarded_nodes
                      << std::endl;
        }
    }

    refresh_ttos_attributes(tos);
    return stats;
}

void flatten_tree_for_support(Tree_of_shapes& tos, long lambda_support)
{
    apply_grain_filter_paper(tos, std::max<long>(1, lambda_support), false, 0);
}

void flatten_tree_for_seed(Tree_of_shapes& tos, long lambda_seed)
{
    apply_grain_filter_paper(tos, std::max<long>(1, lambda_seed), false, 0);
}
