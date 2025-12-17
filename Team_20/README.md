# **Project Title**

Anomaly Detection for Data Leakage Prevention

## *Team Number** 

Team_20

## **Branch**

Computer Science and Engineering (CSE)

 
### Heading 3* **Overview**

This project presents a machine learning–based intrusion detection system designed to identify both known and emerging cyber threats. Traditional signature-based methods often fail against behavioral attacks such as data leakage.

**Methodology**

A weighted ensemble framework is proposed by combining:

- Random Forest

- XGBoost

- Multi-Layer Perceptron (MLP)

- Autoencoder

Feature engineering and Principal Component Analysis (PCA) are used to enhance detection efficiency.

**Dataset**

Training: KDDTrain+

Testing: KDDTest+ (includes novel attacks)

**Results**

Ensemble Accuracy: 69.66%

F1-Score: 0.70

Best Individual Model: XGBoost (72.52% accuracy)

**Conclusion**

The ensemble model improves robustness by balancing precision and anomaly detection, making it more suitable for real-world intrusion detection than single-model approaches.
