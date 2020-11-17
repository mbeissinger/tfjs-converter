# tfjs-converter
Converter from TensorFlow saved model format (1.x) to TensorFlow.js
that lets you specify the output nodes and prunes unused operations.

Made as an interim helper script for Lobe TensorFlow models to be used in TensorFlow.js.

## Setup
Use Python 3.6 or 3.7, make a virtual environment and install the dependencies.

Make the virtual environment
```shell script
python3 -m venv .venv
```
Activate the environment
```shell script
# mac:
source .venv/bin/activate

# windows:
.venv\Scripts\activate
```
Install the requirements
```shell script
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the converter
```shell script
python src/lobe_tf_to_tfjs.py path/to/exported/model --output_dir=path/to/desired/output
```
The first argument to the converter is the path to your exported Lobe TensorFlow saved model.

Optionally, you can give the --output_dir argument of where you would like the TensorFlow.js model to be saved.
If you don't provide this argument, it will be saved inside of your input saved model directory as `tensorflowjs_model/`.


## How to run TensorFlow.js model
Copy the example code directory `example_tfjs_code/` inside of your output TensorFlow.js model directory,
and see the readme there for how to run your model on an image!
