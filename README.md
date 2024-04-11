# tf_Serving


## Prerequisites

- Python 3.x
- Docker (for TensorFlow Serving)
- Operating system: Ubuntu 20.04

### In Hospital Mortality Prediction

- **Description:** This dataset is designed for predicting in-hospital mortality and may include patient records, clinical measurements, and outcomes.

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
2. **run the code**@
    ### In-hospital mortality prediction#
    ##### Logistic Regression
        ```bash
        python in_hospital_mortality/lr/main.py --l2 --C 0.001 --output_dir in_hospital_mortality/lr/  
        ```
    ##### Ann model
        ```bash
        python ptbdb/baseline.py
        ```
    ##### Logistic Regression
        ```bash
        python xgboost/main.py  --output_dir in_hospital_mortality/xgboost/
        ```
    ### Phenotype classification

    ##### LSTM
    Train

    ```
    python phenotyping/rnn/main.py --network models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --output_dir phenotyping/
    ```

    ##### XGBoost
    Train & Test
    ```
    python phenotyping/xgboost/main.py  --output_dir phenotyping/xgboost/
    ```

    ##### LightGBM
    Train & Test
    ```
    python phenotyping/lightgbm/main.py  --output_dir phenotyping/lightgbm/
    ```

    ##### XGBoost FBFE
    Train & Test
    ```
    python phenotyping/xgboost_FBFE/main.py --timestep 1.0  --output_dir phenotyping/xgboost_FBFE/
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








