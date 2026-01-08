<div align="center">
  <h2><b> As-ECTS: Adaptive Shapelet Learning for Early Classification of Streaming Time Series </b></h2>
</div>

[![GitHub Stars](https://img.shields.io/github/stars/Leeway-95/As-ECTS?style=social)](https://github.com/Leeway-95/As-ECTS/stargazers)
![Topic](https://img.shields.io/badge/Streaming%20Time%20Series%20-%20Early%20Classification-blueviolet)

>  ✨ This repository provides the code and datasets for the proposed **As-ECTS.** If you find our work useful for your research, please consider giving it a <strong>star ⭐ on GitHub</strong> to stay updated with future releases.

<!--
## Key Features
InfTS-LLM can be directly applied to any LLMs without retraining:
- ✅ **Native support for multivariate time series**
-->

## Abstract
Early Classification of Time Series (ECTS) aims to predict the class before a sequence is completely observed, which is critical for real-world applications such as time-sensitive medical diagnostics. However, existing ECTS methods typically balance earliness and accuracy under finite-length inputs with limited interpretability for decision-making.
These limitations pose two primary challenges: (1) Adaptive stream modeling; (2) Efficient and interpretable ECTS. 
**As-ECTS** solves the two challenges and achieves competitive performance, which consists of two main components: (a) A Shapelet Similarity Matrix is constructed based on shapelet similarity, where shapelets are critical subsequences to capture distribution changes over streaming time series. This matrix is then incrementally updated through attention-enhanced evaluation for adaptive stream modeling. (b) An Early Shapelet Classifier learns interpretable shapelets and identifies their category through
a shapelet-cache matcher to avoid exhaustive searches over opaque candidates for efficient and interpretable ECTS. 

## Dependencies
* Python 3.10
* numpy 2.2.6
* pandas 2.3.3
* scipy 1.15.3
* scikit-learn 1.7.2
* fastdtw 0.3.4
* pybaobabdt 1.0.1

## Usages
```bash
> git clone https://github.com/Leeway-95/As-ECTS.git
> cd As-ECTS
> conda create -n As-ECTS python=3.10 -y && conda activate As-ECTS && pip install -r requirements.txt
> python main.py
```

## Datasets
1. SmartHome datasets can be obtained from our **datasets directory**.
2. UCR datasets can be downloaded from [UCRArchive_2018.zip](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip).
3. PTB-XL datasets can be downloaded from [PTB-XL-1.0.3.zip](https://physionet.org/content/ptb-xl/get-zip/1.0.3/).

## Contact Us
For inquiries or further assistance, please contact us at [leeway@ruc.edu.cn](mailto:leeway@ruc.edu.cn).
