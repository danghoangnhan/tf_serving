# ECG classification


## Prerequisites

- Python 3.x
- Docker (for TensorFlow Serving)
- Operating system: Ubuntu 20.04

## Dataset
1. The PTB Diagnostic ECG Database
    - Number of Samples: 14552
    - Number of Categories: 2
    - Sampling Frequency: 125Hz
    Data Source: Physionet's PTB Diagnostic Database
## Usage

1. **setup the project:**
   ```bash
   virtualenv venv
   source venv/bin/activate
   ```
2. **run the code**
    ```bash
    python ptbdb/baseline.py
    ```
3. **Deploy into tf_serving:**
    
    ***Download the TensorFlow Serving Docker image***
    ```bash
    docker pull tensorflow/serving
    ```
    ***Location of demo models***
    ```bash
    model="$(pwd)/export_tf/ptpdb"
    ``````
    ***Start TensorFlow Serving container and open the REST API port***
    ```bash
    docker run -t --rm -p 8501:8501 \
        -v "$model:/models" \
        -e MODEL_NAME=cnn_baseline \
        tensorflow/serving &
    ```
4. **Query the model using the predict API**
    ```bash
    curl -d '{"instances": [[1.0, 2.0, 5.0]]}' \
        -X POST http://localhost:8501/v1/models/cnn_baseline:predict
    ```
    **Returns => { "predictions": [2.5, 3.0, 4.5] }**








