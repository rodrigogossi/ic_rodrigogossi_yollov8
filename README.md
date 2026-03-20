# Automatic Fungal Colony Area Measurement Using YOLOv8 Instance Segmentation

> Rodrigo Gossi · Undergraduate Research (*Iniciação Científica*)  
> Repository associated with a manuscript submitted for peer review.

---

## Overview

This repository contains the full experimental pipeline for automatic detection, segmentation, and area measurement of fungal colonies in Petri dish images using YOLOv8 instance segmentation. The system replaces manual measurement workflows (e.g., ImageJ) with a fully automated approach validated against ground-truth measurements.

A physical scale marker (4 cm × 4 cm paper square, 1600 mm²) present in every image is co-detected by the model and used as an in-image ruler, enabling automatic pixel-to-mm² conversion without any additional calibration step.

---

## Key Results

| Metric | YOLOv8n-seg 640px | YOLOv8n-seg 1024px |
|---|---|---|
| **mAP@50** | 0.9942 ± 0.0013 | — |
| **mAP@50-95** | 0.9180 ± 0.0067 | — |
| **R² (vs. ImageJ)** | 0.9984 | 0.9984 |
| **MAPE (vs. ImageJ)** | 6.97% | **5.78%** |
| **Pearson r** | 0.9992 | 0.9992 |
| **Validated samples** | 542 | 542 |

Evaluation was performed using **Repeated K-Fold Cross-Validation (3 repetitions × 5 folds = 15 training runs per resolution)**, ensuring robust and unbiased performance estimation.

---

## Repository Structure

```
ic_rodrigogossi_yollov8/
│
├── dataset_bruto/                        # Raw labeled dataset
│   ├── images/                           # 542 Petri dish images
│   └── labels/                           # YOLO-format polygon annotations
│                                         # (exported from Label Studio)
│
├── resultados_640px/                     # Training outputs — 640px resolution
│   ├── runs/                             # Per-fold model weights and plots
│   │   ├── rep1_fold1/weights/best.pt
│   │   └── ...                           # 15 folds total
│   ├── metricas_completas.csv            # mAP50, mAP50-95, Precision, Recall per fold
│   ├── metricas_por_repeticao.csv        # Mean ± SD per repetition
│   ├── kfold_datasplit.csv               # Train/val assignment per image per fold
│   └── kfold_label_distribution.csv     # Class ratio val/train per fold
│
├── resultados_1024px/                    # Training outputs — 1024px resolution
│   └── ...                              # Same structure as resultados_640px/
│
├── resultados_area_640/                  # Area measurement results — 640px model
│   ├── relatorio_validacao_area.csv      # Per-image: real vs. predicted area + error
│   ├── resumo_estatistico.csv            # R², RMSE, MAE, MAPE, Bland-Altman metrics
│   └── validacao_area_fungica.png        # Scatter + Bland-Altman + residuals figure
│
├── resultados_area_1024/                 # Area measurement results — 1024px model
│   └── ...                              # Same structure as resultados_area_640/
│
├── treinamento_repeated_kfold_640n.ipynb  # Training notebook — 640px
├── treinamento_repeated_kfold_1024n.ipynb # Training notebook — 1024px
├── calculo_de_area_v2_640px.ipynb         # Area measurement notebook — 640px
├── calculo_de_area_v2_1024px.ipynb        # Area measurement notebook — 1024px
├── comparacao_modelos_640_vs_1024.ipynb   # Model comparison figures for paper
├── gabarito.csv                           # Ground-truth area measurements (ImageJ)
└── gerar_gabarito.py                      # Script used to generate gabarito.csv
```

---

## Methods

### Dataset

- **542 images** of fungal colonies growing on Petri dishes
- **2 classes:** `fungo` (fungal colony) and `escala` (physical scale marker)
- Annotated using **Label Studio** with polygon masks, exported in YOLO segmentation format
- Each image contains exactly one scale marker (4 cm × 4 cm paper square) serving as an in-image physical ruler

### Model

- **Architecture:** YOLOv8n-seg (Ultralytics)
- **Training resolutions:** 640 × 640 px and 1024 × 1024 px
- **Pre-trained weights:** `yolov8n-seg.pt` (COCO)
- **Batch size:** 16 | **Epochs:** 100 | **Seed:** 42

### Evaluation — Repeated K-Fold Cross-Validation

```
3 repetitions × 5 folds = 15 independent training runs per resolution
```

Each repetition uses a different random shuffle of the dataset before splitting, following the `RepeatedKFold` implementation from scikit-learn. This approach provides a more reliable estimate of generalization performance compared to standard single-run K-Fold, particularly for small datasets.

### Area Measurement Pipeline

```
Input image
    ↓
YOLOv8-seg inference (retina_masks=True)
    ↓
Extract polygon masks for 'fungo' and 'escala'
    ↓
Compute pixel areas via cv2.contourArea()
    ↓
Conversion factor = 1600 mm² / px_escala
    ↓
Estimated area (mm²) = px_fungo × conversion_factor
    ↓
Compare with ImageJ ground truth
```

### Validation Against ImageJ

Area predictions were compared against manual measurements performed in ImageJ for all 542 images. Statistical validation includes R², RMSE, MAE, MAPE (with 95% bootstrap confidence intervals), Pearson correlation, and Bland-Altman analysis.

---

## Reproducing the Results

### Requirements

```bash
pip install ultralytics scikit-learn pandas numpy matplotlib scipy opencv-python pyyaml tqdm
```

### 1 — Training (Google Colab recommended)

Open `treinamento_repeated_kfold_640n.ipynb` (or `_1024n.ipynb`) in Google Colab.  
Edit **Cell 3** to set `dataset_path` to your Google Drive folder containing `images/` and `labels/`.  
Run all cells in order.

> **Hardware:** T4 GPU + High RAM  
> **Estimated runtime:** ~5–7h (640px) · ~11–12h (1024px)

### 2 — Area Measurement

Open `calculo_de_area_v2_640px.ipynb` (or `_1024px.ipynb`) in Google Colab.  
Edit **Cell 3** to set paths for the trained model (`best.pt`), input images, and `gabarito.csv`.  
Run all cells in order.

### 3 — Model Comparison Figures

Open `comparacao_modelos_640_vs_1024.ipynb` in Google Colab.  
Edit **Cell 4** to set paths for the result CSVs.  
Run all cells. Six publication-ready figures will be saved to your Drive.

---

## Classes and Annotation Format

Annotations follow the **YOLO segmentation format** (polygon per instance):

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

| Class ID | Name | Description |
|---|---|---|
| 0 | `escala` | Physical scale marker (4 cm × 4 cm paper square) |
| 1 | `fungo` | Fungal colony |

---

## Ground Truth (gabarito.csv)

Manual area measurements performed in ImageJ for all 542 images.

| Column | Description |
|---|---|
| `identificador` | Image filename without extension |
| `Area(mm²)` | Manually measured fungal colony area in mm² |

---

## Generated Figures (comparacao_modelos_640_vs_1024.ipynb)

| File | Description |
|---|---|
| `fig1_dispersao_comparativa.png` | Scatter plot: predicted vs. real area (both resolutions) |
| `fig2_bland_altman_comparativo.png` | Bland-Altman agreement analysis |
| `fig3_distribuicao_erros.png` | Error distribution: violin + box plot + histogram |
| `fig4_kfold_metricas_comparativo.png` | Segmentation metrics across all 15 folds |
| `fig5_radar_comparativo.png` | Radar chart: overall performance profile |
| `fig6_tabela_comparativa.png` | Summary table with best-model highlights |
| `fig7_mAP5095_todos_folds.png` | mAP@50-95 for all 30 trained models |
| `fig8_ranking_30_modelos.png` | Ranked comparison of all 30 models |

---

## Citation

> *Article under review. Citation will be updated upon publication.*

---

## License

This repository is made available for reproducibility purposes in association with the submitted manuscript. The dataset and results may not be redistributed without permission from the authors.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Label Studio](https://labelstud.io/) — annotation platform
- [scikit-learn RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html)
- [ImageJ](https://imagej.net/) — ground-truth area measurements
