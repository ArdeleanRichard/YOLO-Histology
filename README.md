# Object detection in histology 

*A multi-dataset benchmark and test-time inference*

This repository contains the code and configurations for a study submitted as a scientific research article.

<!-- 
[![DOI](https://img.shields.io/badge/DOI-10.3390/diagnostics15141823-blue)](https://doi.org/10.3390/diagnostics15141823) 

This repository contains the code and configurations used in the study titled:
**"Can YOLO Detect Retinal Pathologies? A Step Towards Automated OCT Analysis"**
by Adriana-Ioana Ardelean, Eugen-Richard Ardelean, and Anca Marginean, published in *Diagnostics*, 2025.
-->

## 📄 Overview

This project benchmarks modern object detection models (YOLOv8–v12, YOLOE, YOLO-World, RT-DETR) for Histopathology analysis. It evaluates their ability to detect retinal biomarkers across two publicly available datasets:

* **BCNB**,
* **Nuclei**,
* **TNBC**,
* **MoNuSAC**,
* **CryoNuSeg**,

Our experiments identify **YOLOE** and **YOLOv11** as the most effective models, offering an excellent trade-off between detection accuracy and computational efficiency.

Moreover, we propose Test-time Graph Similarity Propagation, a new method for test-time inference that improves upon the traditional confidence threshold and the TBSP algorithm.


---

## 📊 Datasets

This project requires a .yaml file to be created for the models to run on each dataset, as shown in this example:
```
train:  ..\..\data\MoNuSAC\images\train
val:    ..\..\data\MoNuSAC\images\val
test:   ..\..\data\MoNuSAC\images\test

nc: 4  # number of classes
names: ['Epithelial', 'Lymphocyte', 'Neutrophil', 'Macrophage']  # class names
```

### 1. BCNB Dataset

* **Name**: BCNB
* **Description**: digitized H&E-stained WSIs of primary tumor core-needle biopsy (CNB) specimens for predicting axillary lymph node (ALN) metastasis in early breast cancer (EBC). This extensive collection includes data from 1,058 EBC patients with clinically negative ALN, enrolled between May 2010 and August 2020.
* **Access**: Openly available.
  🔗 [BCNB Dataset Page](https://bcnb.grand-challenge.org/)

### 2. Nuclei Dataset

* **Name**: Nuclei
* **Description**: 143 2,000 x 2,000 images ER+ BCa images scanned at 40x with 12,000 nuclei manually segmented across 137 patients. 
* **Access**: Openly available.
  🔗 [Nuclei Dataset Page](https://andrewjanowczyk.com/use-case-1-nuclei-segmentation/)

### 3. TNBC Dataset

* **Name**: TNBC
* **Description**: 50 images from 11 patients with a total of 4,022 annotated cells from H&E stained histological slides at 40x magnification. These slides were sourced from Triple Negative Breast Cancer (TNBC) patients
* **Access**: Openly available.
  🔗 [TNBC Dataset Page](https://zenodo.org/records/2579118#.Yt5FWt_RaUk)

### 4. MoNuSAC Dataset

* **Name**: MoNuSAC
* **Description**: multi-organ dataset specifically designed for nuclei segmentation in histopathological images to encourage the computer vision research community to develop and test algorithms for detecting, segmenting, and classifying nuclei, tasks crucial for characterizing the tumor microenvironment (TME) in cancer prognostication and research. This large and diverse dataset comprises over 46,000 hand-annotated nuclei, sourced from 71 patients across 31 hospitals. It includes nuclei from four different organs and four distinct nucleus types: epithelial cells, lymphocytes, macrophages, and neutrophils. 
* **Access**: Openly available.
  🔗 [MoNuSAC Dataset Page](https://monusac-2020.grand-challenge.org/)

### 5. CryoNuSeg Dataset

* **Name**: CryoNuSeg
* **Description**: manually annotated nuclei instance segmentation dataset derived exclusively from frozen section (FS) H&E-stained images. CryoNuSeg comprises images from 10 human organs that were not exploited in other publicly available datasets, specifically the adrenal gland, larynx, lymph node, mediastinum, pancreas, pleura, skin, testis, thymus, and thyroid gland. It includes 30 whole slide images (WSIs), from which 512x512 pixel image patches acquired at 40x magnification were extracted, with one patch per WSI. 
* **Access**: Openly available.
  🔗 [CryoNuSeg Dataset Page](https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images)



---

## 🧠 Models Evaluated

* YOLOv8 to YOLOv12
* YOLOE: Prompt-guided detection
* YOLO-World: Open-vocabulary object detection
* RT-DETR: Transformer-based object detection



## 📈 Results Summary

| Inferencer   | Model   | Dataset  | F1 Score  | 
| -------------| ------- | -------  | --------- | 
| Conf>0.5     | RT-DETR | TNBC     | 0.000     | 
| TSBP 		   | RT-DETR | TNBC     | 0.000     | 
| TGSP         | RT-DETR | TNBC     | **0.355** | 
| Conf>0.5     | YOLOE   | nuclei   | 0.182     | 
| TSBP 		   | YOLOE   | nuclei   | 0.330     | 
| TGSP         | YOLOE   | nuclei   | **0.456** | 
| Conf>0.5     | YOLOv11 | nuclei   | 0.791     | 
| TSBP 		   | YOLOv11 | nuclei   | 0.790     | 
| TGSP         | YOLOv11 | nuclei   | **0.878** | 

---
<!-- 
## 📜 Citation

If you use this code or reference the models/datasets in your work, please cite:

```bibtex
@article{Ardelean2025YOLO,
  title     = {Can YOLO Detect Retinal Pathologies? A Step Towards Automated OCT Analysis},
  author    = {Ardelean, Adriana-Ioana and Ardelean, Eugen-Richard and Marginean, Anca},
  journal   = {Diagnostics},
  year      = {2025},
  volume    = {15},
  number    = {14},
  pages     = {1823},
  doi       = {10.3390/diagnostics15141823}
}
```

---
-->

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---

