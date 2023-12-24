# tf_Serving


## Prerequisites

- Python 3.x
- Docker (for TensorFlow Serving)
- Operating system: Ubuntu 20.04

### Phenotyping

- **Description:** This dataset contains information related to phenotyping, which may include patient characteristics, medical history, and other relevant data.

- **Source:** Kaggle

- **Data Format:** CSV

- **Data Fields:** [List and describe the key fields or columns in the dataset. You can find this information on the Kaggle dataset page.]

- **Data Usage:** This dataset can be used for various tasks related to phenotyping analysis, machine learning, and data science.

- **License:** Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/uzair54/phenotyping) for information regarding the dataset's license and terms of use.

- **Download Link:** You can download the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/uzair54/phenotyping).

### In Hospital Mortality Prediction

- **Description:** This dataset is designed for predicting in-hospital mortality and may include patient records, clinical measurements, and outcomes.

- **Source:** Kaggle

- **Data Format:** [Specify the format of the dataset, e.g., CSV, JSON.]

- **Data Fields:** [List and describe the key fields or columns in the dataset. You can find this information on the Kaggle dataset page.]

- **Data Usage:** This dataset can be used for predicting in-hospital mortality, healthcare analytics, and machine learning research.

- **License:** Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction) for information regarding the dataset's license and terms of use.

- **Download Link:** You can download the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction).

### Usage Guidelines

[Include any specific guidelines or instructions for users who want to use the dataset, such as data preprocessing steps or ethical considerations.]




## Usage

1. **setup the project:**
   ```bash
   virtualenv venv
   source venv/bin/activate
   ```
2. **run the code**
    ### In-hospital mortality prediction
    1. Logistic Regression
        ```bash
        python in_hospital_mortality/lr/main.py --l2 --C 0.001 --output_dir in_hospital_mortality/lr/  
        ```
    2. Ann model
        ```bash
        python ptbdb/baseline.py
        ```
    3. Logistic Regression
        ```bash
        python xgboost/main.py  --output_dir in_hospital_mortality/xgboost/
        ```
3. **Deploy into tf_serving:**
    
    ***Download the TensorFlow Serving Docker image***
    ```bash
    docker pull tensorflow/serving
    ```
    ***Location of demo models***
    ```bash
    model="$(pwd)/export_tf/moritality"
    ``````
    ***Start TensorFlow Serving container and open the REST API port***
    ```bash
    docker run -t --rm -p 8501:8501 \
        -v "$model:/models" \
        -e MODEL_NAME=ann_baseline \
        tensorflow/serving &
    ```
4. **Query the model using the predict API**
    ```bash
    curl -d '{"instances": [[1.0, 2.0, 5.0]]}' \
        -X POST http://localhost:8501/v1/models/cnn_baseline:predict
    ```
    **Returns => { "predictions": [2.5, 3.0, 4.5] }**








