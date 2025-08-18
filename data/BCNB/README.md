### BCNB datasets
Paper: https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.759007/full
Data: https://bcnb.grand-challenge.org/
Data: https://bupt-ai-cz.github.io/BCNB/


### Early Breast Cancer Core-Needle Biopsy WSI Dataset
Motivation
Breast cancer (BC) has become the greatest threat to women’s health worldwide. Clinically, identification of axillary lymph node (ALN) metastasis and other tumor clinical characteristics such as ER, PR, and so on, are important for evaluating the prognosis and guiding the treatment for BC patients.

Several studies intended to predict the ALN status and other tumor clinical characteristics by clinicopathological data and genetic testing score. However, due to the relatively poor predictive values and high genetic testing costs, these methods are often limited. Recently, deep learning (DL) has enabled rapid advances in computational pathology, DL can perform high-throughput feature extraction on medical images and analyze the correlation between primary tumor features and above status. So far, there is no relevant research on preoperatively predicting ALN metastasis and other tumor clinical characteristics based on WSIs of primary BC samples.

### Dataset
We have introduced a new dataset of Early Breast Cancer Core-Needle Biopsy WSI (BCNB), which includes core-needle biopsy whole slide images (WSIs) of early breast cancer patients and the corresponding clinical data. The WSIs have been examined and annotated by two independent and experienced pathologists blinded to all patient-related information.

There are WSIs of 1058 patients, and only part of tumor regions are annotated in WSIs. Except for the WSIs, we have also provided the clinical characteristics of each patient, which includes age, tumor size, tumor type, ER, PR, HER2, HER2 expression, histological grading, surgical, Ki67, molecular subtype, number of lymph node metastases, and the metastatic status of ALN.


### Research and Task
This is an educational challenge, so we do not limit the specific content for your research, and any research based on the BCNB Dataset is welcome.

For your convenience in research, we have split the BCNB Dataset into training cohort, validation cohort, and independent test cohort with the ratio as 6: 2: 2, and the following tasks are feasible based on the BCNB Dataset:

1. The prediction of the metastatic status of ALN, including N0, N+(1-2), and N+(>2).
2. The prediction of the histological grading, including 1, 2 and 3.
3. The prediction of molecular subtype, including Luminal A, Luminal B, Triple Negative and HER2(+).
4. The prediction of HER2, including positive and negative.
5. The prediction of ER, including positive and negative.
6. The prediction of PR, including positive and negative.
7. The aforementioned tasks are all binary or multiple classification, and the metrics for evaluation could be AUC, ACC, SENS, SPEC, PPV and NPV. If you have some new experimental results, you can publish them here.

By the way, the "Task 1: the prediction of the metastatic status of ALN" has been studied in our paper titled “Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides”, which has provided a baseline performance with multiple instance learning (MIL), please visit the github repo for more details.



### Citation
If you have used the BCNB Dataset in your research, please cite the paper that introduced the BCNB Dataset:

  @article{xu2021predicting,
    title={Predicting Axillary Lymph Node Metastasis in Early Breast Cancer Using Deep Learning on Primary Tumor Biopsy Slides},
    author={Xu, Feng and Zhu, Chuang and Tang, Wenqi and Wang, Ying and Zhang, Yu and Li, Jie and Jiang, Hongchuan and Shi, Zhongyue and Liu, Jun and Jin, Mulan},
    journal={Frontiers in Oncology},
    pages={4133},
    year={2021},
    publisher={Frontiers}
  }


### Access
Dear scholars,

Thanks for your mail.

Our data can be downloaded through the following link:

Google Drive: https://drive.google.com/drive/folders/1HcAgplKwbSZ7ZZl2m6PZdvVF70QJmVuR?usp=sharing

OneDrive: https://bupteducn-my.sharepoint.com/:f:/g/personal/tangwenqi_bupt_edu_cn/EoFPuyWfkg1OpYw_QKK7aBUBKBGSazefo4qlRnaBnCtmKA?e=XPQJqh

Aliyun Drive: https://www.aliyundrive.com/s/qyLGbvDY5uk

Baidu Yun: https://pan.baidu.com/s/1wJfeadl_vw6bKnt-G6ykFA Password: n7cs

Considering the larger size of WSIs files, please download them in multiple batches to prevent the corruption of zip files, especially for OneDrive.

If you have problems regarding our data, please email to: czhu@bupt.edu.cn, tangwenqi@bupt.edu.cn

We appreciate your interest in our work and data again.

Our dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms in https://github.com/bupt-ai-cz/BALNMP#license

For more information, please refer to our github: https://github.com/bupt-ai-cz or CVSM Group: https://teacher.bupt.edu.cn/zhuchuang/en/index.htm

Welcome to continue to pay attention to our follow-up updates.

Best wishes,

CVSM Group, AI, BUPT.

