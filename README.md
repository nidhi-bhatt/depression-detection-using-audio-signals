# Depression Detection Using Audio Signals

## Overview

This project uses machine learning to detect depression from audio recordings of participants' speech. The model analyzes acoustic features extracted from speech segments to classify individuals as depressed or non-depressed.

## Features

- **Audio Processing**: Speech segmentation and acoustic feature extraction using openSMILE's eGeMAPSv02 feature set
- **Data Augmentation**: Time stretching and pitch shifting to enhance model robustness
- **Balanced Learning**: SMOTE oversampling to address class imbalance
- **Ensemble Approach**: Voting classifier combining Decision Trees, Random Forests, Gradient Boosting, and KNN models
- **Performance**: Optimized for recall to minimize false negatives in depression detection

## Results

The final ensemble model achieves:

- Accuracy: 78.69%
- Precision: 77.78%
- Recall: 75%
- F1 Score: 76.36%

## Implementation

The model pipeline includes:

1. Audio preprocessing and participant speech segmentation
2. Feature extraction using openSMILE
3. Data augmentation for depressed class samples
4. Feature scaling and class balancing
5. Ensemble model training and optimization
6. Model evaluation with focus on recall metrics

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, librosa, opensmile, scikit-learn, imbalance-learn

## Future Work

- Explore deep learning approaches (CNNs, RNNs)
- Incorporate linguistic features from transcripts
- Develop real-time depression screening tools
