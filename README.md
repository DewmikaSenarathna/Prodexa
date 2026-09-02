# PRODEXA

**PROD**uct **EX**amination & **A**nalysis — image processing based system to detect, classify, and generate statistical summaries
of supermarket products from images — developed for **EC9570: Digital Image Processing**,
Department of Computer Engineering, University of Jaffna.

## Overview

Given an image of a supermarket basket / product layout, the system:
1. Preprocesses the image (noise removal, resizing, normalization)
2. Detects and segments individual products
3. Classifies each product into a category
4. Generates a statistical summary report (counts, percentages, charts)
5. Displays results as labeled bounding boxes on the image and/or console output

Real-time capture is not used — the system works on pre-collected / manually captured images.
All processing is performed **locally** (no cloud APIs).

## Dataset

**[RPC: Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)**
(Wei et al., 200 SKUs across 17 meta-categories, ~83,739 images, COCO-format bounding box
annotations, ~20GB). Two image types:

- **Exemplar images** — single product, clean/plain background, multiple viewing angles.
  Used for training and validating the classification module.
- **Checkout images** — multiple products placed on a checkout tray, three clutter levels
  (easy: 3–5 categories / 3–10 instances, medium, hard). Used for testing detection,
  segmentation, and the full end-to-end pipeline.

**Category strategy:** classification targets the dataset's **17 meta-categories**
(e.g. bottle-like, box-like, canister-like, bag-like) rather than all 200 fine-grained SKUs —
these are visually distinguishable by shape/color/texture and better suited to a classical
image-processing pipeline within the project's scope.

**Scope for this project (12-hour limit):** a curated subset is used, not the full dataset —
a sample of exemplar images per meta-category for training/testing classification, and
**easy-difficulty** checkout images for testing detection/segmentation and the full pipeline
demo (higher clutter levels involve heavy occlusion, which conflicts with the assignment's
"clearly separated products preferred" scope).

> The dataset (~20GB) is **not stored in this repository** — see [Getting the Dataset](#getting-the-dataset) below.

## Team

| Member   | Modules Owned                                      |
|----------|-----------------------------------------------------|
| De Costa M.S.M.     | Image Acquisition & Preprocessing, Product Classification |
| Senarathna S.A.D.H. | Object detection and segmentation,Report Generation       |

## Pipeline

```
Raw Image → Preprocessing → Segmentation → Classification → Statistics & Report → Annotated Output
```


## Project Structure

```
prodexa/
├── data/
│   ├── raw/                  # RPC dataset subset (exemplar + checkout images) — git-ignored
│   ├── annotations/          # COCO-format bounding box JSON (from RPC dataset) — git-ignored
│   ├── processed/            # preprocessed images (Member A output)
│   ├── dataset/               # cropped product images organized by meta-category, for classifier training
│   └── dataset_subset.txt    # list of exact files used from the full RPC dataset (for reproducibility)
├── notebooks/                # Jupyter notebooks, one per pipeline stage
├── src/                       # reusable Python modules imported by notebooks
├── models/                    # trained classifier artifacts
├── outputs/
│   ├── annotated_images/     # bounding boxes + labels
│   └── reports/               # summary tables / charts
└── docs/                       # demo notes for viva
```

> `data/raw/`, `data/annotations/`, and `data/dataset/` are listed in `.gitignore` — only
> `dataset_subset.txt` (a manifest, not the images themselves) is version-controlled, so
> teammates/graders can reproduce the exact subset used without a 20GB repo.



## Techniques Used

**Preprocessing**
- Resizing, Gaussian/median blur, color space conversion, lighting normalization 

**Segmentation**
- Thresholding , HSV-based background masking, morphological operations,
  contour detection

**Classification**
- _(fill in once decided: handcrafted features + ML classifier, or pretrained CNN feature
  extractor + shallow classifier — with justification, since pretrained models must be
  explained as required by the assignment)_

**Statistics**
- Total product count, category-wise counts, percentage distribution, bar/pie charts

## Setup

```bash
git clone <repo-url>
cd prodexa
pip install -r requirements.txt
jupyter notebook
```

## Getting the Dataset

The dataset is **not included in this repository** (too large for GitHub). To set it up locally:

1. Download from Kaggle: [Retail Product Checkout Dataset](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)
   (requires a free Kaggle account), either manually or via the Kaggle API:
   ```bash
   kaggle datasets download -d diyer22/retail-product-checkout-dataset
   ```
2. Extract into `data/raw/` (this path is git-ignored).
3. This project uses a **curated subset** of the full dataset (see [Dataset](#dataset) above) —
   the specific images/annotations used for training and testing are listed in
   `data/dataset_subset.txt` (or regenerated via a setup script — to be added).

## Usage

1. Place input images in `data/raw/`
2. Run notebooks in order (`01` → `02` → `03` → `04`), or run
   `00_pipeline_integration.ipynb` for the full end-to-end pipeline
3. Outputs (annotated images, reports, charts) are saved to `outputs/`

## Results

| Metric | Value |
|--------|-------|
| Classification accuracy | _TBD (target ≥ 80%)_ |
| Categories | _TBD_ |
| Test images | _TBD_ |

## Contribution Workflow



## Limitations

- Assumes single or clearly separated products (no heavy overlap)
- Local processing only; no cloud-based inference
- Accuracy validated on a limited, manually curated dataset

## Course Info

EC9570 – Digital Image Processing
Department of Computer Engineering, Faculty of Engineering, University of Jaffna
