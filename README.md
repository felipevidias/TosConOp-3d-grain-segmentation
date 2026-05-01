# TosConOp-3d-grain-segmentation

## Overview

This repository is a C++ fork of Julien Mendes Forte’s **ToSConOp** project, extended to support **3D grayscale volumetric TIFF stacks** and an experimental **grain-oriented analysis pipeline** based on the **Tree of Shapes (ToS)** and **connected operators**.

The project has two complementary goals:

1. **ToS / ToSConOp core**  
   Build, simplify, and modify a **3D Tree of Shapes** using connected-operator principles inspired by recent ToSConOp work.

2. **3D grain application layer**  
   Use the **same filtered ToS hierarchy** to:
   - extract **bright volumetric grain supports**,
   - detect **dark internal seed candidates**,
   - generate support labels,
   - generate seed markers,
   - produce final grain labels on a **3D crop**.

This fork is intended as an **experimental research codebase** for 3D granular-material volumes, not as a finished command-line package.

---

## Scientific background

This repository is related to three main lines of work:

- the classical **Tree of Shapes** introduced by Caselles, Coll, and Morel;
- the recent **connected operators on trees of shapes** developed by Julien Mendes Forte, Nicolas Passat, Akinobu Shimizu, and Yukiko Kenmochi;
- recent work on **hierarchical analysis of 3D X-ray CT granular images** by Lysandre Macke, Yukiko Kenmochi, Nicolas Passat, and collaborators.

In this fork:

- the **Tree of Shapes / ToSConOp layer** is used to construct and simplify the hierarchy;
- the **grain support and seed extraction** are implemented as an application layer on top of that filtered hierarchy.

---

## Current methodological status

The current implementation is centered on a **single 3D Tree of Shapes hierarchy**.

In other words:

- **bright grain supports** and
- **dark internal seed candidates**

are both extracted from the **same filtered ToS**, rather than from two separate trees or from an external segmentation stage.

This is an important point: the code is currently organized around **one ToS**, with selective ToSConOp-style filtering applied to different types of leaf structures inside that same hierarchy.

---

## What this fork adds compared with the original project

Compared with the original 2D-oriented proof-of-concept, this fork adds:

- support for **3D volumetric TIFF stacks** loaded with OpenCV;
- construction of a **3D Tree of Shapes** through Higra;
- reconstruction of **3D filtered volumes** from the modified tree;
- a **grain-analysis pipeline** that includes:
  - support extraction,
  - seed extraction,
  - marker generation,
  - final grain labeling;
- quality-control outputs for visual inspection of seeds and labeled grains;
- an example workflow on a **200×200×200 crop**.

---

## Current implementation status

At the moment, the repository contains two main layers.

### 1. ToS / ToSConOp core

The core includes:

- `Tree_of_shapes`
- `Node_tos`
- tree preprocessing
- node enrichment
- area computation
- altitude modification
- 3D reconstruction
- selective leaf filtering inspired by ToSConOp

This is the part directly related to the **Tree of Shapes hierarchy** and **connected-operator behavior**.

### 2. 3D grain application layer

The application layer includes:

- support candidate extraction from bright ToS structures,
- seed detection from dark internal ToS structures,
- support labels,
- marker volumes,
- final grain labels,
- QC images and CSV summaries.

This layer is **application-specific** and should not be described as a literal implementation of every step from the ToSConOp papers. It is better understood as a **grain-analysis workflow built on top of a filtered ToS hierarchy**.

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
Higra is used to build the **3D Tree of Shapes** hierarchy.

### xtensor / xtl
These are required because the Higra ToS construction uses xtensor-based image containers.

---

## Suggested directory layout

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
