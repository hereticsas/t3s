# T3S: TensorFlow Simple Stupid Server
### Overview
The **T3S** is an open-source project to make the development of **Flask REST APIs using a TensorFlow model** easier. The goal is to quickly setup an API to which you pass input data in the URL, that computes predictions using the TensorFlow model and outputs the results.

With this API, you can pass one or multiple examples to your model at once. It works with a simple HTTP request-response system. When the API is asked to process a given input, it parses the URL according to certain rules (see below) then it calls the model to make a prediction and answers back with the result (either numerical or categorical).

Although it was originally designed for email analysis by the HERETIC SAS company, the aim is to make T3S as standard as possible. In the end, it would be for everyone who is interested in linking a TensorFlow model to a web API and, hopefully, those willing to contribute to its improvement!

The T3S is programmed in Python 3.

### Configuring the server
First make sure you have a model saved somewhere in a `${TF_MODEL_DIR}` directory. To learn how to prepare and export a model, you can check out the [TensorFlow reference 'Wide and Deep' model tutorial](https://www.tensorflow.org/tutorials/wide_and_deep).

Then to install dependencies pass `-r requirements.txt` to pip (i.e. run `pip3 install --user -r requirements.txt`).

Finally, edit the `config.py` file to suit your needs:
1. set the `${SERVER_NAME}` to the address and port of your API in the form: `{@server:port}` (e.g. `127.0.0.1:5000`)
2. choose your server configuration: `default`, `dev`, `testing` or `production` and
set it in the `configure_app()` function
3. configure your TensorFlow model by setting the `${TF_MODEL_DIR}` directory
4. specify your feature computing mode:

The T3S is primarily designed only for model prediction and not feature computing, meaning you can pre-process your data and extract your feature values in whatever you wish, then send them to the API as a JSON-formatted string.
To use this raw mode, you need to set the `TF_EXTRACTOR` variable to `None`.

However, if you would rather keep it all in the same place, you can also provide a custom features extractor. This extractor should be a class that inherits from the `T3T3SExtractor` abstract class and holds specific implementations of the `check_data()` and `compute_features()` functions.
*Note: the `extract()` function should not be touched since it runs the process independently from your model.*
To use this personalized mode, you need to set the `TF_EXTRACTOR` variable to your custom extractor class.

### Running the server
To start the server, simply run: `python api.py`.

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

depending on your configuration file: this page will print out the prediction results of your model for each given example. The output is a JSON dictionary that either shows the prediction foreach example, or contains a single 'error' key with a message explaining the issue.

*Note: the T3S is not meant to do pretty-formatting: results are simply outputted in the page without any styling.*

### Known Issues & Perspectives
Despite our best efforts, it is complex to make an API adapted to any type of TensorFlow model. Datatypes processing, in particular, could probably be improved. For now, inputs are converted based on their Python variable type but there is no check to insure they match the types request by your model.

### Development History

- v1.0: initial version - Spring 2018

- v1.1: multiple changes - June 2018:

    - added configuration and project setup files

    - standardized data types

    - added in-app Python features extraction

    - wrote the first version of the documentation

### A step-by-step setup with the 'Wide and Deep' model

To give a concrete example of a T3S complete set up, we will consider the tutorial 'Wide and Deep' (W&D) TensorFlow model and configure a T3S in raw mode to compute predictions with this model.

##### Model export

First, we need to export the W&D model so we can use it in our API. To do so, as explained [in the TensorFlow tutorial](https://www.tensorflow.org/tutorials/wide_and_deep), you need to:

- download [TensorFlow's official models](https://github.com/tensorflow/models)

- add this model folder to your Python path: `export PYTHONPATH="$PYTHONPATH:/path/to/models"`

- install dependencies by passing `-r official/requirements.txt` to pip. (i.e. `pip3 install --user -r official/requirements.txt`)

- go to the `official/wide_deep` model folder

- run `python data_download.py` to download the data (in the `/tmp/census_data` folder by default; you can change the destination with the `--data-dir` flag)

- run `python wide_deep.py --export-dir ${TF_MODEL_DIR}` to train the model and export it to the `${TF_MODEL_DIR}` directory

You should now have a model saved in the export folder as a file: `${TF_MODEL_DIR}/${TIMESTAMP}/saved_model.pb` (where `${TIMESTAMP}` is automatically specified by the system, e.g. `1524249124`).

##### T3S configuration

Now, we can configure the T3S to use this model. After downloading the file, you should edit the `config.py` file:

- let's leave the default for the `${SERVER_NAME}` variable: `127.0.0.1:5000`

- likewise, we will keep the default `dev` configuration

- set the `${TF_MODEL_DIR}` variable to the folder where you saved your TensorFlow model (with the `${TIMESTAMP}` part, e.g. `/tmp/wide_deep_model/1524249124/`)

- make sure you are in raw mode, i.e. `TF_EXTRACTOR = None`
*(see below for custom extractor configuration)*

##### T3S running

Finally, run: `python api.py`.

To test that your API is working, you can check out the following address:
`127.0.0.1:5000/[{"age":46.0, "education_num":10.0, "capital_gain":7688.0, "capital_loss":0.0, "hours_per_week":38.0}, {"age":24.0, "education_num":13.0, "capital_gain":0.0, "capital_loss":0.0, "hours_per_week":50.0}]`

You should get a JSON response containing two lines with the predicted features for these two examples, for example:
```
{
    "ex0-res": 0.9594689011573792,
    "ex1-res": 0.1530081182718277
}
```

##### Bonus: Custom mode configuration

To use a custom extractor, you must write a class that inherits from the `T3SExtractor` abstract class. You need to specify `check_data()` and `compute_features()` function that match your model. For instance, here is a valid custom extractor to work on email addresses:

```
from t3s import T3SExtractor

class CustomExtractor(T3SExtractor):

    def check_data(self, input):
        return '@' in input

    def compute_features(self, input):
        # compute features values
        lp, domain = input.split('@')
        lp_length = len(lp)
        lp_num = sum(c.isdigit() for c in lp)
        lp_alpha = sum(c.isalpha() for c in lp)
        lp_other = lp_length - lp_num - lp_alpha

        # create features dictionary
        features = {
            'lp_length': lp_length,
            'lp_alpha': lp_alpha,
            'lp_num': lp_num,
            'lp_other': lp_other,
            'domain_length': len(domain),
            'domain': domain,
        }

        return features

    def error_formatting(self):
        return 'Please enter emails in the form: "username@domain".'
```

*Note 1: to know what type of values each function should return, refer to the `TF_EXTRACTOR` abstract class in the `t3s.py` file.*

*Note 2: the `error_formatting()` function is not essential but if implemented, it will provide a more precise error message to the user if the input is not formatted right.*

Then you have to set this class as value for the configuration `TF_EXTRACTOR` variable. So, if your class is in the `extractor.py` file in your T3S folder, you should edit `config.py` and set:

`TF_EXTRACTOR = __import__("custom").CustomExtractor()`

You are now in custom mode and, for instance, you can access the address:

`127.0.0.1:5000/example@ex.com;example2@ex2.com`

from which features will be computed and used to make predictions with an email analysis model.

**Warning:** be careful, when in custom mode, raw input in the JSON format will be considered invalid (unless your `check_data()` implementation handles it).
