# T3S: Tensorflow Simple Stupid Server
## Overview
The **T3S** is an open-source project to make the development of Flask REST APIs using a TensorFlow model
easier. The goal is to quickly setup an API at which you pass a string as input data, that computes a
prediction using the TensorFlow model and outputs the result.

---

?

## Setup
First make sure you have a model saved somewhere in a `${TF_MODEL_DIR}` directory.

Then, edit the `config.py` to suit your needs:

### Running the code
To start the server, simply run:

```
python api.py
```

This will run the server at the `${SERVER_NAME}` address and port specified in your configuration file.

You can now ask your model to predict outputs for given data by passing it in the JSON form as URL. For
example, you can access the page: `${SERVER_NAME}/{"feature1": value1, "feature2": value2}`.
This will print the prediction result of your model.
