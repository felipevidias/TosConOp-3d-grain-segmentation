#pragma once

#include <limits>
#include <vector>

struct Node_tos;

struct GeoStats {
    int minx = std::numeric_limits<int>::max();
    int miny = std::numeric_limits<int>::max();
    int minz = std::numeric_limits<int>::max();

    int maxx = std::numeric_limits<int>::min();
    int maxy = std::numeric_limits<int>::min();
    int maxz = std::numeric_limits<int>::min();

    int dx = 0;
    int dy = 0;
    int dz = 0;

    long volume = 0;
    int num_slices = 0;

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;

    bool touches_border = false;
};

struct ZTrackStats {
    int z_min_active = 0;
    int z_max_active = 0;
    int active_slices = 0;
    int longest_run = 0;

    int z_peak = 0;
    int area_peak = 0;

    double continuity_score = 0.0;
    double area_smooth_score = 0.0;
    double centroid_smooth_score = 0.0;
    double track_score = 0.0;
};

struct LocalStats {
    double center_mean = 0.0;
    double shell_mean = 0.0;
    double contrast = 0.0;
};

struct SupportCandidate {
    Node_tos* node = nullptr;
    GeoStats geo;
    ZTrackStats track;

    long area = 0;
    long alt = 0;
    long parent_alt = 0;
    int delta_parent = 0;

    double mean_intensity = 0.0;
    double fill_ratio = 0.0;
    double score = 0.0;
};

struct FinalGrain {
    int label = 0;
    long node_id = -1;

    std::vector<long> voxels;
    GeoStats geo;
    ZTrackStats track;

    long area = 0;
    long alt = 0;
    long parent_alt = 0;
    int delta_parent = 0;

    double mean_intensity = 0.0;
    double fill_ratio = 0.0;
    double score = 0.0;

    int seed_x = -1;
    int seed_y = -1;
    int seed_z = -1;
    int seed_value = 255;

    bool seed_from_dark_candidate = false;
};
