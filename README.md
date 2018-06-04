# T3S: Tensorflow Simple Stupid Server
## Overview
The **T3S** is an open-source project to make the development of **Flask REST APIs using a TensorFlow model** easier.
The goal is to quickly setup an API at which you pass input data in the URL, computes a prediction using the TensorFlow
model and outputs the result.

It was originally conceived for email analysis, this is why most examples refer to this type of data.

---

For now, input data can be given in two forms:
- a JSON-formatted string that holds the **pre-computed features values** of your example
(e.g. *{"lp_length": 7, "domain": "ex.com"}*)
- a string to extract the features from thanks to your **specific extracting file**
(e.g. *example&#64;ex.com*)

The **specific extracting file** must be saved as `tf/extractor.py` in your T3S folder. It has to implement an `extract()`
function that returns a Python dictionary with each of your model features as keys and the matching values for your example.

*Note: checking if the input string is in the right form should be done here and the* `extract()` *function should return* `None` *if there is an error.*

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

*127.0.0.1:5000/{"lp_length": 7, "domain": "ex.com"}*

or

*127.0.0.1:5000/example&#64;ex.com*

depending on your configuration file: this page will print out the prediction result of your model.
