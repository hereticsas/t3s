# T3S: Tensorflow Simple Stupid Server
## Overview
The **T3S** is an open-source project to make the development of **Flask REST APIs using a TensorFlow model** easier. The goal is to quickly setup an API to which you pass input data in the URL, that computes predictions using the TensorFlow model and outputs the results.

With this API, you can pass one or multiple examples to your model at once.

It was originally conceived for email analysis, this is why most examples refer to this type of data.

### Configuring the server
First make sure you have a model saved somewhere in a `${TF_MODEL_DIR}` directory. To learn how to prepare and export a model, you can check out the [TensorFlow reference ‘Wide and Deep’ model tutorial](https://www.tensorflow.org/tutorials/wide_and_deep).

Then, edit the `config.py` file to suit your needs:
1. set the `${SERVER_NAME}` to the address and port of your API in the form: `{@server:port}` (e.g. `127.0.0.1:5000`)
2. choose your server configuration: `default`, `dev`, `testing` or `production` and
set it in the `configure_app()` function
3. configure your TensorFlow model by setting the `${TF_MODEL_DIR}` directory
4. specify your feature computing mode:

The T3S is primarily designed only for model prediction and not feature computing, meaning you can pre-process your data and extract your feature values in whatever you wish, then send them to the API as a JSON-formatted string.
However, if you would rather keep it all in the same place, you can also edit the `tf/extractor.py` file in the T3S folder. You will need to modify the `check_data()` and `compute_features()` functions to adapt them to your model. The `extract()` function should not be touched since it runs the process independently from your model.

### Running the server
To start the server, simply run:

```
python api.py
```

This will run the server at the `${SERVER_NAME}` address and port specified in your configuration file.

You can now ask your model to predict outputs for given data by passing it in the URL
in the JSON format or as a string.

### Processing some data
Broadly speaking, you will access an address in the form: `${SERVER_NAME}/data_input`.

For now, `data_input` can be given in two forms:

- a JSON-formatted string that holds the **pre-computed features values** of your examples in an array of JSON dictionaries
(e.g. `[{"lp_length": 7, "domain": "ex.com"},{"lp_length": 4, "domain": "ex2.com"}]`)
- a string (with the examples separated by the ``;`` character) to extract the features from thanks to your **specific extracting file**
(e.g. `example@ex.com;example2@ex2.com`)

For example, you can access the page:

`http://127.0.0.1:5000/[{"lp_length": 7, "domain": "ex.com"}]`

or

`http://127.0.0.1:5000/example@ex.com`

depending on your configuration file: this page will print out the prediction results of your model for each given example.
