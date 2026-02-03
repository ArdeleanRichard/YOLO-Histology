# Comprehensive Analysis Report

## 1. Object Size Category Analysis

### Summary Statistics

| size_category   |   ('precision', 'mean') |   ('precision', 'std') |   ('recall', 'mean') |   ('recall', 'std') |   ('f1', 'mean') |   ('f1', 'std') |
|:----------------|------------------------:|-----------------------:|---------------------:|--------------------:|-----------------:|----------------:|
| large           |                0        |              0         |             0        |           0         |         0        |       0         |
| medium          |                0.30174  |              0.0293955 |             0.972156 |           0.0238233 |         0.460107 |       0.0370936 |
| small           |                0.198088 |              0.0558111 |             0.892442 |           0.0724165 |         0.31875  |       0.0602523 |
| tiny            |                0        |              0         |             0        |           0         |         0        |       0         |

### Best Performing Models by Size Category

**Tiny objects:**
- RTDETR: F1=0.000
- YOLO8: F1=0.000
- YOLO9: F1=0.000

**Small objects:**
- RTDETR: F1=0.449
- YOLO11: F1=0.356
- YOLO12: F1=0.316

**Medium objects:**
- YOLOE: F1=0.510
- YOLOW: F1=0.479
- YOLO10: F1=0.476

**Large objects:**
- RTDETR: F1=0.000
- YOLO8: F1=0.000
- YOLO9: F1=0.000

## 2. Statistical Significance Analysis

Performed 28 pairwise comparisons

### Top Performance Differences

- **RTDETR vs YOLO10**: 