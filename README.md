## GitHub Repository Description

### Project: Secure Decentralized Hybrid Model for ECG Data Processing and Prediction

This repository contains the implementation of a secure decentralized hybrid model designed to process and predict ECG (Electrocardiogram) image and time series data. The project addresses the data shortage problem in cardiology by creating a secure and generalized algorithm capable of identifying various types of cardiological issues.

### Objectives

1. **Creating a Decentralized and Secure Model Training Environment for ECG Data**
2. **Developing a Hybrid Model Architecture to Process and Predict ECG Image and Time Series Data**

### Dataset Details

![image](https://github.com/RishikGupta28/AI-for-Cardiology/assets/74090072/dca4eb71-4296-432a-8db1-61e830e41870)


The ECG data used in this project were obtained using a non-commercial PTB prototype recorder with the following specifications:
- **Channels**: 16 input channels (14 for ECGs, 1 for respiration, 1 for line voltage)
- **Input Voltage**: ±16 mV, compensated offset voltage up to ± 300 mV
- **Input Resistance**: 100 Ω (DC)
- **Resolution**: 16 bit with 0.5 μV/LSB (2000 A/D units per mV)
- **Bandwidth**: 0 - 1 kHz (synchronous sampling of all channels)
- **Noise Voltage**: max. 10 μV (pp), 3 μV (RMS) with input short circuit
- **Additional Features**: Online recording of skin resistance and noise level during signal collection

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `tensorflow` and `keras`: For building and training the hybrid model.
  - `opencv-python`: For image processing tasks.
  - `numpy` and `pandas`: For numerical computations and data manipulation.
  - `matplotlib` and `seaborn`: For data visualization.

### Algorithm and Approach

1. **ECG Image to Binary Image Conversion**:
   ![image](https://github.com/RishikGupta28/AI-for-Cardiology/assets/74090072/50bbceda-96cc-4477-8cb5-24d2d46c805e)

    - **Description**: Convert the ECG image into a binary image, transforming each pixel to either black or white.
    - **Purpose**: Simplifies the image, emphasizing the essential features of the ECG signal and making subsequent processing more straightforward.

![image](https://github.com/RishikGupta28/AI-for-Cardiology/assets/74090072/a2f39ce7-a002-478b-9bcd-9cee18a1f2c6)

3. **Defining Variable H**:
    - **Description**: H is defined as the vertical resolution of the binary image, representing the total number of vertical pixels.
    - **Purpose**: This resolution is used later to normalize the amplitude measurements of the ECG signal.

4. **Creating the Sample Values Vector V**:
    - **Description**: Sample the amplitude values of the ECG signal at regular intervals along the horizontal axis, defined by variable S.
    - **Purpose**: Ensures capturing the entire waveform of the ECG signal.

5. **Defining Variable S**:
    - **Description**: S specifies the horizontal distance between consecutive sampling points.
    - **Purpose**: By sampling at these regular intervals, a comprehensive representation of the ECG signal is captured.

6. **Actual and Predicted Data**:
    - **Description**: The actual sampled values from the ECG signal are denoted as Yactual.
    - **Normalization**: To normalize these sampled values, each h value is divided by H (the vertical resolution of the image).
### Result
![image](https://github.com/RishikGupta28/AI-for-Cardiology/assets/74090072/d4a15855-20de-4745-bf55-31c4f65991c0)

### Usage

To run the project locally:
1. Clone the repository.
2. Install the required libraries using `requirements.txt`.
3. Follow the steps to generate your own dataset or use the provided dataset.
4. Train the model using the provided Jupyter notebooks and scripts.
5. Test and evaluate the model performance using the provided scripts.

### Example Code Snippets

#### Importing Required Libraries
```python
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
```

#### ECG Image to Binary Image Conversion
```python
def convert_to_binary(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image
```

#### Sampling ECG Signal
```python
def sample_ecg_signal(binary_image, S):
    h, w = binary_image.shape
    H = h
    V = []
    for x in range(0, w, S):
        column = binary_image[:, x]
        h_value = np.min(np.where(column == 0))
        V.append(h_value / H)
    return V
```

#### Normalizing Sample Values
```python
def normalize_values(V, H):
    return [v / H for v in V]
```

### Conclusion

This repository provides a comprehensive implementation of a secure decentralized hybrid model for ECG data processing and prediction. The project aims to address data shortages and create a generalized algorithm capable of identifying various cardiological problems. By leveraging both image and time series data, this hybrid model enhances the accuracy and reliability of ECG-based diagnostics.
