#pragma once

#include <cstddef>
#include <string>

struct TtosGrainParams {
    // Preserve more grain-level branches than the 95-grain baseline.
    long dark_leaf_area_max = 16;
    long bright_leaf_area_max = 14;

    long dark_max_delta_parent = 24;
    long bright_max_delta_parent = 24;

    long dark_max_interval_amplitude = 34;
    long bright_max_interval_amplitude = 34;

    bool show_operator_progress = true;
    std::size_t operator_progress_every = 200;
    std::size_t operator_refresh_every = 512;

    long support_area_min = 8;
    long support_area_max = 180000;

    double support_tree_score_min = 0.022;
    double support_seed_descendant_score_min = 0.0002;
    std::size_t max_support_candidates_keep = 100000;
    std::size_t max_final_grains = 12000;

    // Keep promotion off to avoid swallowing smaller supports.
    int support_promotion_max_steps = 0;
    double support_promotion_score_margin = 0.02;
    double support_promotion_area_ratio_max = 1.60;

    // Seed selection fully from MIN_TREE descendants.
    long seed_area_min = 1;
    long seed_area_max = 28000;
    int seed_min_delta_parent = 1;

    int max_seed_descendant_depth = 110;
    bool allow_support_fallback_seed = false;

    std::string default_input_tif =
        "/home/felipe/Pesquisa-Grain_Seg/Filtro_Conexo/data/processed/EFRGP01_00_roi_core.tif";
};
