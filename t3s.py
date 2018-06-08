from flask_restful import Resource

import tensorflow as tf
import numpy as np
import json

import os

from tensorflow.python.tools import saved_model_utils
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.client import session
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader
from tensorflow.python.debug.wrappers import local_cli_wrapper

import config

class T3S(Resource):

    def get(self, data_input):
        """
        Processes the given input to predict results from the TensorFlow model
        for one or multiple examples.

        If the T3S is configured with a features extraction file, the input should
        contain all examples separated by a ';' character. Each of these substrings
        must be in the right format for the extract() function defined in the
        extractor file.

        Otherwise, the input must be a JSON-formatted string representing an array
        of JSON dictionaries with the features of the model as keys, and the
        pre-computed features values foreach example.

        Args:
            input: String containing the examples to process.

        Returns:
            A dictionary that contains the prediction results.
        """
        # If no features extraction file is given, expect direct JSON data
        if config.TF_EXTRACTOR is None:
            try:
                inputs = json.loads(data_input)
            except json.decoder.JSONDecodeError:
                return {
                    'error': '"%s" is not valid data. Please enter JSON-formatted data to represent your features.' % (data_input)
                }

            if not isinstance(inputs, list):
                inputs = [inputs]
        # Else expect a string and extract features with the extracting file
        else:
            inputs = config.TF_EXTRACTOR.extract(data_input)
            if None in inputs:
                return {
                    'error': '"%s" is not valid data. ' % (data_input) + \
                        config.TF_EXTRACTOR.error_formatting()
                }

        # Cast and process examples
        json_result = {}
        for i, parsed_json in enumerate(inputs):
            model_input = T3S.preprocess_input_examples_arg_string('examples=['+json.dumps(parsed_json)+']')
            feature_chance = T3S.run_saved_model_with_feed_dict(config.TF_MODEL_DIR, "serve", "predict", model_input, './', True)
            json_result['ex' + str(i) + '-res'] = np.float64(feature_chance)

        return json_result


    def run_saved_model_with_feed_dict(saved_model_dir, tag_set, signature_def_key,
                                       input_tensor_key_feed_dict, outdir,
                                       overwrite_flag, tf_debug=False):
      """Runs SavedModel and fetch all outputs.
      Runs the input dictionary through the MetaGraphDef within a SavedModel
      specified by the given tag_set and SignatureDef. Also save the outputs to file
      if outdir is not None.

      Args:
        saved_model_dir: Directory containing the SavedModel to execute.
        tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
            string format, separated by ','. For tag-set contains multiple tags, all
            tags must be passed in.
        signature_def_key: A SignatureDef key string.
        input_tensor_key_feed_dict: A dictionary that maps input keys to numpy ndarrays.
        outdir: A directory to save the outputs to. If the directory doesn't exist,
            it will be created.
        overwrite_flag: A boolean flag to allow overwrite output file if file with
            the same name exists.
        tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the
            intermediate Tensor values and runtime GraphDefs while running the
            SavedModel.

      Returns:
        An strings with the computed prediction for the input.

      Raises:
        ValueError: When any of the input tensor keys is not valid.
        RuntimeError: An error when output file already exists and overwrite is not
            enabled.
      """
      result_string = ''

      # Get a list of output tensor names.
      meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)

      # Re-create feed_dict based on input tensor name instead of key as session.run
      # uses tensor name.
      inputs_tensor_info = T3S._get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)

      # Check if input tensor keys are valid.
      for input_key_name in input_tensor_key_feed_dict.keys():
        if input_key_name not in inputs_tensor_info.keys():
          raise ValueError(
              '"%s" is not a valid input key. Please choose from %s, or use '
              '--show option.' %
              (input_key_name, '"' + '", "'.join(inputs_tensor_info.keys()) + '"'))

      inputs_feed_dict = {
          inputs_tensor_info[key].name: tensor
          for key, tensor in input_tensor_key_feed_dict.items()
      }
      # Get outputs
      outputs_tensor_info = T3S._get_outputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key)
      # Sort to preserve order because we need to go from value to key later.
      output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
      output_tensor_names_sorted = [
          outputs_tensor_info[tensor_key].name
          for tensor_key in output_tensor_keys_sorted
      ]

      with session.Session(graph=ops_lib.Graph()) as sess:
        loader.load(sess, tag_set.split(','), saved_model_dir)

        if tf_debug:
          sess = local_cli_wrapper.LocalCLIDebugWrapperSession(sess)

        outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)

        for i, output in enumerate(outputs):
          output_tensor_key = output_tensor_keys_sorted[i]
          if output_tensor_key == "probabilities" :
            feature_chance = output[0][1]
            result_string = feature_chance

          # Only save if outdir is specified.
          if outdir:
            # Create directory if outdir does not exist
            if not os.path.isdir(outdir):
              os.makedirs(outdir)
            output_full_path = os.path.join(outdir, output_tensor_key + '.npy')

            # If overwrite not enabled and file already exist, error out
            if not overwrite_flag and os.path.exists(output_full_path):
              raise RuntimeError(
                  'Output file %s already exists. Add \"--overwrite\" to overwrite'
                  ' the existing output files.' % output_full_path)

            np.save(output_full_path, output)

        return result_string

    @staticmethod
    def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                    signature_def_key):
      """Gets TensorInfo for all inputs of the SignatureDef.
      Returns a dictionary that maps each input key to its TensorInfo for the given
      signature_def_key in the meta_graph_def

      Args:
        meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to
            look up SignatureDef key.
        signature_def_key: A SignatureDef key string.

      Returns:
        A dictionary that maps input tensor keys to TensorInfos.
      """
      return signature_def_utils.get_signature_def_by_key(meta_graph_def,
                                                          signature_def_key).inputs

    @staticmethod
    def _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                     signature_def_key):
      """Gets TensorInfos for all outputs of the SignatureDef.
      Returns a dictionary that maps each output key to its TensorInfo for the given
      signature_def_key in the meta_graph_def.

      Args:
        meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefmap to
        look up signature_def_key.
        signature_def_key: A SignatureDef key string.

      Returns:
        A dictionary that maps output tensor keys to TensorInfos.
      """
      return signature_def_utils.get_signature_def_by_key(meta_graph_def,
                                                          signature_def_key).outputs

    @staticmethod
    def preprocess_input_examples_arg_string(input_examples_str):
        """Parses input into dict that maps input keys to lists of tf.Example.
        Parses input string in the format of 'input_key1=[{feature_name:
        feature_list}];input_key2=[{feature_name:feature_list}];' into a dictionary
        that maps each input_key to its list of serialized tf.Example.

        Args:
            input_examples_str: A string that specifies a list of dictionaries of
            feature_names and their feature_lists for each input.
            Each input is separated by semicolon. For each input key:
                'input=[{feature_name1: feature_list1, feature_name2:feature_list2}]'
            items in feature_list can be the type of float, int, long or str.

        Returns:
            A dictionary that maps input keys to lists of serialized tf.Example.

        Raises:
            ValueError: An error when the given tf.Example is not a list.
        """
        input_dict = T3S.preprocess_input_exprs_arg_string(input_examples_str)
        for input_key, example_list in input_dict.items():
          if not isinstance(example_list, list):
            raise ValueError(
                'tf.Example input must be a list of dictionaries, but "%s" is %s' %
                 (example_list, type(example_list)))
          input_dict[input_key] = [
               T3S._create_example_string(example) for example in example_list
          ]
        return input_dict

    @staticmethod
    def preprocess_input_exprs_arg_string(input_exprs_str):
      """Parses input arg into dictionary that maps input key to python expression.
      Parses input string in the format of 'input_key=<python expression>' into a
      dictionary that maps each input_key to its python expression.

      Args:
        input_exprs_str: A string that specifies python expression for input keys.
            Each input is separated by semicolon. For each input key:
                'input_key=<python expression>'
      Returns:
        A dictionary that maps input keys to their values.
      Raises:
        RuntimeError: An error when the given input string is in a bad format.
      """
      input_dict = {}

      for input_raw in filter(bool, input_exprs_str.split(';')):
        if '=' not in input_exprs_str:
          raise RuntimeError('--input_exprs "%s" format is incorrect. Please follow'
                             '"<input_key>=<python expression>"' % input_exprs_str)
        input_key, expr = input_raw.split('=', 1)
        # ast.literal_eval does not work with numpy expressions
        input_dict[input_key] = eval(expr)  # pylint: disable=eval-used
      return input_dict

    @staticmethod
    def _create_example_string(example_dict):
      """Creates a serialized tf.example from feature dictionary.

      Args:
        example_dict: The dictionary that contains the example features.

      Returns:
        A byte-string to represent the serialized example data.
      """
      # Cast features in TensorFlow types
      features = {}
      for f_name, f_val in example_dict.items():
          features[f_name] = T3S._cast_feature(f_val)

      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=features))
      # Serialize to string
      return example.SerializeToString()

    @staticmethod
    def _cast_feature(value):
        """
        Casts a value to a TensorFlow type depending on its Python type.

        Args:
            value: Value to cast.
                Can be: float, str, int or bytes.

        Returns:
            A tf.train.Feature with matching type.
        """
        if isinstance(value, float):
            return T3S._float_feature(value)
        elif isinstance(value, str):
            return T3S._bytes_feature(tf.compat.as_bytes(value))
        elif isinstance(value, int):
            return T3S._int64_feature(value)
        elif isinstance(value, bytes):
            return T3S._bytes_feature(value)
        else:
            raise ValueError(
                'Type %s for value %s is not supported for tf.train.Feature.' %
                (type(value), value)
            )

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class T3SExtractor(object):
    """Abstract class to inherit from to create a specific features extractor
    adapted to your model"""

    def check_data(self, input):
        """
        Checks for the input validity.

        Args:
            input: String to check.

        Returns:
            True if the input is valid for the model, False otherwise.
        """
        raise NotImplementedError('You need to provide a specific data '
            'checking function matching your model.')

    def compute_features(self, input):
        """
        Computes the TensorFlow model features from the input.

        Args:
            input: String to compute the features from.

        Returns:
            A dictionary with the model's keys as keys and the computed features as
            values.
        """
        raise NotImplementedError('You need to provide a specific extracting '
            'function matching your model.')

    def extract(self, data, debug=False):
        """
        Runs the features extraction process.

        Args:
            data: String to compute the features from (can contain multiple examples,
                each separated by a ';' character).
            debug: A boolean flag to display or not the computed features.

        Returns:
            If the input data is not in the right form, returns None.
            Else, returns a dictionary with the model's keys as keys and the computed
            features as values.
        """
        # Split examples
        inputs = data.split(';')

        if debug:
            print('=' * 23)
            print('T3S computation result:')
            print('=' * 23)

        examples_features = []
        # Compute each example
        for input in inputs:
            features = None

            if self.check_data(input):
                features = self.compute_features(input)

                if debug:
                    print('Example #%3d:' % len(examples_features))
                    print('-' * 13)
                    print('Input:', input)
                    print('Result:', features, '\n')

            examples_features.append(features)

        return examples_features

    def error_formatting(self):
        """
        Informs the user of the correct input format for this features extractor.
        Null by default, should be overriden.
        """
        return ''
