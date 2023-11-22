# ECG classification


## Prerequisites

- Python 3.x
- Docker (for TensorFlow Serving)
- Operating system: Ubuntu 20.04

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
    model="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"
    ``````
    ***Start TensorFlow Serving container and open the REST API port***
    ```bash
    docker run -t --rm -p 8501:8501 \
        -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
        -e MODEL_NAME=half_plus_two \
        tensorflow/serving &
    ```
4. **Query the model using the predict API**
```bash
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
```
**Returns => { "predictions": [2.5, 3.0, 4.5] }**








