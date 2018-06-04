# T3S: Tensorflow Simple Stupid Server
## Overview
The **T3S** is an open-source project to make the development of **Flask REST APIs using a TensorFlow model** easier. The goal is to quickly setup an API to which you pass input data in the URL, that computes predictions using the TensorFlow model and outputs the results.

With this API, you can pass one or multiple examples to your model at once.

It was originally conceived for email analysis, this is why most examples refer to this type of data.

---

For now, input data can be given in two forms:
- a JSON-formatted string that holds the **pre-computed features values** of your examples in an array of JSON dictionaries
(e.g. *[{"lp_length": 7, "domain": "ex.com"},{"lp_length": 4, "domain": "ex2.com"}]*)
- a string (with the examples separated by the ``;``) character to extract the features from thanks to your **specific extracting file**
(e.g. *example@ex.com;example2@ex2.com*)

The **specific extracting file** must be saved as `tf/extractor.py` in your T3S folder. It implements an `extract()` function that returns an array of Python dictionaries with each of your model features as keys and the matching values for each example.

*Note: as mentioned in the `tf/extractor.py` file, you need to adapt its functions `check_data()` and `compute_features()` so that they match your model. On the other hand, the `extract()` function should not be touched, it runs the process independently from your model features.*

### Configuring the server
First make sure you have a model saved somewhere in a `${TF_MODEL_DIR}` directory.

Then, edit the `config.py` file to suit your needs:
1. set the `${SERVER_NAME}` to the address and port of your API in the form: `{@server:port}` (e.g. `127.0.0.1:5000`)
2. choose your server configuration: `default`, `dev`, `testing` or `production` and
set it in the `configure_app()` function
3. configure your TensorFlow model by setting the `${TF_MODEL_DIR}` directory and,
if necessary, specifying you have a features extraction file

### Running the server
To start the server, simply run:

```
python api.py
```

This will run the server at the `${SERVER_NAME}` address and port specified in your configuration file.

You can now ask your model to predict outputs for given data by passing it in the URL
in the JSON format or as a string.
For example, you can access the page:

*127.0.0.1:5000/[{"lp_length": 7, "domain": "ex.com"},{"lp_length": 4, "domain": "ex2.com"}]*

or

*127.0.0.1:5000/example@ex.com;example2@ex2.com*

depending on your configuration file: this page will print out the prediction results of your model for each given example.
