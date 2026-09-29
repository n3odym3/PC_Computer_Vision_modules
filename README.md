# Computer Vision Modules for Pipeline Creator

A collection of computer vision, image processing, segmentation, and algorithmic analysis modules for [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator).

This repository contains computer vision modules (barcode and QR decoding, AI-assisted interactive segmentation, color classification, transparent image extraction, and flat-field optical correction) designed for plug-and-play integration into the Pipeline Creator ecosystem.

---

## Installation

To install and use one or more modules in **Pipeline Creator**:

1. **Install Module Dependencies**:

   Install the common dependencies and any module-specific packages within your Pipeline Creator environment:
   ```bash
   pip install opencv-python numpy loguru
   # Install module-specific packages as needed:
   pip install pyzbar                     # High-speed barcode/QR decoding (Barcode_reader)
   pip install torch segment-anything      # Deep learning interactive segmentation (SAM)
   ```

2. **Copy Modules to Pipeline Creator**:
   
   Copy and paste the folder(s) of the desired module(s) (such as `Barcode_reader`, `Color_mapper`, `Image_extractor`, `SAM`, `Vignetting_corrector`, etc.) directly into the `modules/` folder of your **Pipeline Creator** installation:
   ```
   Pipeline_creator/
   ├── modules/
   │   ├── Barcode_reader/
   │   ├── Color_mapper/
   │   ├── Image_extractor/
   │   ├── SAM/
   │   ├── Vignetting_corrector/
   │   └── ...
   ```

3. **Launch Pipeline Creator**:

   The copied modules will be automatically detected and made available in the Pipeline Creator node editor.

---

## Available Modules

| Module | Description | Key Dependencies |
| :--- | :--- | :--- |
| [**Barcode_reader**](./Barcode_reader/) | Real-time detection and decoding of 1D barcodes and 2D QR codes with change detection, automatic format normalization, and downstream text dispatching. Falls back to OpenCV when `pyzbar` is not present. | `pyzbar` (or OpenCV fallback), `opencv-python`, `numpy`, `loguru` |
| [**Color_mapper**](./Color_mapper/) | Classifies pixels across full-frame or masked regions by Euclidean/CIE-Lab distance to a reference color palette. Features an interactive palette calibration editor, live ROI probing, real-time class distribution stats, and cropped transparent RGBA export. | `opencv-python`, `numpy`, `loguru` |
| [**Image_extractor**](./Image_extractor/) | Isolates segmented subjects from `(Frame, Mask)` pairs, sets background pixels to transparent alpha (`0`), and copies the resulting RGBA image directly to the Windows system clipboard (supporting PNG alpha & DIB formats). | `opencv-python`, `numpy`, `loguru` |
| [**SAM**](./SAM/) | Interactive deep learning segmentation powered by Meta's Segment Anything Model (SAM). Supports positive/negative prompt points (`CTRL+Click`), image pre-processing (gamma, saturation), GPU/CUDA acceleration, and outputs masks and synchronized `(Frame, Mask)` pairs. | `torch`, `segment-anything`, `opencv-python`, `numpy`, `loguru` |
| [**Vignetting_corrector**](./Vignetting_corrector/) | Rectifies radial lens illumination falloff and optical shading artifacts by applying flat-field calibration matrices. Supports 8-bit, 12-bit, and 16-bit depth processing with real-time before/after comparison mode. | `opencv-python`, `numpy` |

---
