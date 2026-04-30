#pragma once

#include <cstddef>
#include <string>

struct TtosGrainParams {
    long lambda_support = 15;
    long lambda_seed = 2;

    long support_area_min = 120;
    long support_area_max = 130000;

    int support_dx_min = 4;
    int support_dy_min = 4;
    int support_dz_min = 3;

    int support_dx_max = 95;
    int support_dy_max = 95;
    int support_dz_max = 190;

    int min_active_slices = 3;
    int min_longest_z_run = 3;

    double support_score_min = 0.95;

    std::size_t max_support_candidates_keep = 9000;
    std::size_t max_final_grains = 3000;

    double support_center_nms = 9.0;
    double support_bbox_overlap_nms = 0.30;

    long seed_area_min = 1;
    long seed_area_max = 5000;
    int seed_min_delta_parent = 2;

    int seed_dx_max = 45;
    int seed_dy_max = 45;
    int seed_dz_max = 140;

    std::string default_input_tif =
        "/home/felipe/Pesquisa-Grain_Seg/Filtro_Conexo/data/processed/EFRGP01_00_roi_core.tif";
};
