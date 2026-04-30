# TosConOp-3d-grain-segmentation

## Overview

This repository is a C++ fork of Julien Mendes Forte’s **ToSConOp** project, extended to support **3D grayscale volumetric TIFF stacks** and a **grain-oriented analysis pipeline** based on the **Tree of Shapes (ToS)** and **connected operators**.

The project has two complementary goals:

1. **ToS / ToSConOp core**  
   Build and modify a **Tree of Shapes** using connected-operator principles inspired by the recent ToSConOp literature.

2. **3D grain application layer**  
   Use the filtered ToS hierarchy to:
   - extract **bright volumetric grain supports**,
   - detect **dark internal seed candidates**,
   - assign fallback seeds when necessary,
   - produce support labels, seed markers, and final grain labels on a **3D crop**.

This fork is intended as an **experimental research codebase** for 3D granular-material volumes, not as a finished command-line tool.

---

## Scientific background

This fork is related to three main lines of work:

- the classical **Tree of Shapes** image model introduced by Caselles, Coll, and Morel;  
- the recent **connected operators on trees of shapes** developed by Julien Mendes Forte, Nicolas Passat, and Yukiko Kenmochi;  
- recent work on **hierarchical analysis of 3D X-ray CT granular images** by Lysandre Macke, Yukiko Kenmochi, and Nicolas Passat. citeturn696089search2turn616674search4turn696089search6turn283994search2

In this repository, the **ToSConOp-like filtering** is used in the tree simplification / flattening stage, while the **3D grain support and seed extraction** are implemented as an application layer on top of the filtered ToS hierarchy.

---

## What this fork adds compared with the original project

Compared with the original 2D-oriented proof-of-concept, this fork adds:

- support for **3D volumetric TIFF stacks** loaded with OpenCV;
- construction of a **3D Tree of Shapes** through Higra;
- reconstruction of **3D filtered volumes** from the modified tree;
- a **grain pipeline** that separates:
  - support extraction,
  - seed extraction,
  - marker generation,
  - final grain labeling;
- quality-control outputs for visual inspection of seeds and final grain labels;
- an example workflow on a **200×200×200 crop**.

---

## Current implementation status

At the moment, the repository contains:

### 1. ToS / ToSConOp core
The core includes:
- `Tree_of_shapes`
- `Node_tos`
- tree preprocessing
- node enrichment
- area computation
- altitude modification
- 3D reconstruction

This is the part directly related to the Tree of Shapes hierarchy and connected-operator behavior.

### 2. 3D grain application layer
The grain layer includes:
- support candidate extraction from bright ToS structures,
- seed detection from dark internal ToS structures,
- support-based fallback seed assignment,
- support labels,
- marker volumes,
- final grain labels,
- QC images and CSV summaries.

This layer is **application-specific** and is not claimed to be a literal implementation of every step of the ToSConOp papers.

---

## Dependencies

You will need:

- **CMake** ≥ 3.14
- **C++20** compiler
- **OpenCV**
- **Higra**
- **xtensor**
- **xtl**

### OpenCV
OpenCV is used to:
- read grayscale TIFF stacks,
- write TIFF outputs,
- generate QC images.

### Higra
Higra is used to build the **Tree of Shapes** hierarchy.

### xtensor / xtl
These are required because the Higra ToS construction uses xtensor-based image containers.

---

## Suggested directory layout

The project is expected to follow a structure similar to:

```text
TosConOp-3d-grain-segmentation/
├── CMakeLists.txt
├── include/
│   ├── node_ct.h
│   ├── node_tos.h
│   ├── tree_of_shapes.h
│   ├── tree_of_shapes_edit.h
│   ├── ttos_grain_analysis.h
│   ├── ttos_grain_params.h
│   ├── ttos_grain_pipeline.h
│   ├── ttos_grain_qc.h
│   └── ttos_grain_types.h
├── src/
│   ├── main.cpp
│   ├── node_ct.cpp
│   ├── node_tos.cpp
│   ├── tree_of_shapes.cpp
│   ├── tree_of_shapes_edit.cpp
│   ├── ttos_grain_analysis.cpp
│   ├── ttos_grain_pipeline.cpp
│   └── ttos_grain_qc.cpp
└── build/
```

---

## CMakeLists.txt

A suitable `CMakeLists.txt` for this fork is:

```cmake
cmake_minimum_required(VERSION 3.14)
project(TosConOp3DGrainSegmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-O3 -ffast-math -g -DNDEBUG)

add_executable(edit_tos
    src/main.cpp
    src/tree_of_shapes.cpp
    src/node_tos.cpp
    src/tree_of_shapes_edit.cpp
    src/ttos_grain_analysis.cpp
    src/ttos_grain_pipeline.cpp
    src/ttos_grain_qc.cpp
    src/node_ct.cpp
)

target_include_directories(edit_tos
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    /path/to/Higra/include
    /path/to/xtensor/include
    /path/to/xtl/include
)

find_package(OpenCV REQUIRED)
target_include_directories(edit_tos PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(edit_tos PRIVATE ${OpenCV_LIBS})
```

Replace:
- `/path/to/Higra/include`
- `/path/to/xtensor/include`
- `/path/to/xtl/include`

with the actual paths on your machine.

---

## Build instructions

From the project root:

```bash
mkdir build
cd build
cmake ..
make -j4
```

If the configuration succeeds, the executable will be:

```bash
./edit_tos
```

---

## Input data

The expected input is a **3D grayscale TIFF stack**.

Typical example:

```bash
./edit_tos /path/to/volume.tif
```

The current `main.cpp` also supports explicitly providing crop coordinates:

```bash
./edit_tos /path/to/volume.tif x0 y0 z0
```

where:
- `x0` = crop origin along X
- `y0` = crop origin along Y
- `z0` = crop origin along Z

If coordinates are not provided, the code searches for a suitable **off-center crop** of size **200×200×200**.

---

## Current execution flow

The current example in `main.cpp` follows this general sequence:

1. Load the grayscale TIFF stack.
2. Select a **200×200×200 crop**.
3. Compute intensity percentiles on the crop.
4. Build a **support Tree of Shapes**.
5. Filter / flatten the support tree.
6. Reconstruct the filtered support volume.
7. Extract support candidates from bright ToS structures.
8. Build a **seed Tree of Shapes**.
9. Filter / flatten the seed tree.
10. Reconstruct the filtered seed volume.
11. Detect seed candidates from dark internal ToS structures.
12. Assign fallback seeds when dark seeds are missing.
13. Build:
    - support labels,
    - seed markers,
    - final grain labels.
14. Write TIFF / PNG / CSV outputs.

---

## Main output files

The current `main.cpp` produces outputs such as:

- `input_gray_stack_8u.tif`  
  Input crop.

- `support_reconstructed_8u.tif`  
  Reconstructed support-filtered volume.

- `seed_reconstructed_8u.tif`  
  Reconstructed seed-filtered volume.

- `support_labels_16u.tif`  
  Label image of accepted grain supports.

- `grain_markers_points_16u.tif`  
  Marker volume containing one seed per accepted grain.

- `final_grain_labels_16u.tif`  
  Final grain labels.

- `grain_tracks_qc_rgb.tif`  
  RGB QC visualization of labeled grains.

- `seed_orthoview_qc.png`  
  Orthogonal-view seed visualization.

- `seed_list.csv`  
  CSV summary of final seeds.

- `metadata.txt`  
  Summary of crop coordinates, percentiles, counts, and runtime.

---

## Interpreting the outputs

A practical order for inspection is:

1. `grain_tracks_qc_rgb.tif`  
   Quick visual sanity check.

2. `final_grain_labels_16u.tif`  
   Main segmentation result.

3. `grain_markers_points_16u.tif`  
   Check whether the seeds are correctly placed.

4. `seed_orthoview_qc.png`  
   Fast 2D orthoview summary.

5. `seed_list.csv`  
   Tabular seed information.

---

## Important note about methodology

This repository combines:

- a **ToS / ToSConOp-inspired hierarchical filtering core**, and
- a **3D grain-analysis pipeline** built on top of that hierarchy.

Therefore:

- the **tree construction and tree filtering stages** are the parts most directly connected to the ToS / ToSConOp literature;
- the **support scoring, NMS, seed selection, fallback seeds, and final grain labeling** are application-driven extensions for 3D granular image analysis.

This distinction is important when describing the code in a paper, report, or email.

---

## Limitations

Current limitations include:

- the code is still **research-oriented**;
- the command line is still minimal;
- full-volume execution may be expensive in time and memory;
- current experiments are more stable on **cropped 3D subvolumes**;
- some segmented grains may still show noisy boundaries;
- seed detection is currently stronger than the final segmentation quality.

---

## Repository origin

This project is based on Julien Mendes Forte’s original ToSConOp proof-of-concept and extends it toward 3D volumetric grain analysis.

If you are using this fork in an academic context, please make clear the distinction between:

- the original ToSConOp concepts and implementation ideas,
- the 3D extensions introduced here,
- and the grain-specific application logic.

---

## References

1. V. Caselles, B. Coll, and J.-M. Morel, “Topographic maps and local contrast changes in natural images,” *International Journal of Computer Vision*, vol. 33, no. 1, pp. 5–27, 1999.

2. J. Mendes Forte, N. Passat, A. Shimizu, and Y. Kenmochi, “Consistent connected operators based on trees of shapes,” *SIAM Journal on Imaging Sciences*, vol. 18, no. 4, pp. 2547–2579, 2025. citeturn616674search4

3. J. Mendes Forte, N. Passat, and Y. Kenmochi, “Designing connected operators using the topological tree of shapes,” in *Discrete Geometry and Mathematical Morphology (DGMM 2025)*, pp. 195–209, 2025, doi: 10.1007/978-3-032-09544-2_15. citeturn616674search1turn616674search4

4. N. Passat, J. Mendes Forte, Y. Kenmochi, et al., “A topological tree of shapes,” in *Discrete Geometry and Mathematical Morphology*, 2022. citeturn696089search2turn696089search6

5. L. Macke, Y. Kenmochi, and N. Passat, “Hierarchical analysis of 3D X-ray CT images for granular materials,” 2024. citeturn616674search2turn616674search8

6. B. Perret, G. Chierchia, J. Cousty, S. J. F. Guimarães, Y. Kenmochi, et al., “Higra: Hierarchical Graph Analysis,” *SoftwareX*, vol. 10, 100335, 2019. citeturn283994search2turn283994search13
