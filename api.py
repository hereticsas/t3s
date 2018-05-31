from flask import Flask
from flask_restful import Resource, Api

import tensorflow as tf
import os
import numpy as np
import json

from tensorflow.core.example import example_pb2
from tensorflow.python.tools import saved_model_utils
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.client import session
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.saved_model import loader

app = Flask(__name__)
api = Api(app)
dir_model = 'tf/models/official/wide_deep/email_predictor_model/1527447753'


class T3S(Resource):

    def get(self,data_input):
        #data_input = '{"lp_length":10, "lp_alpha":9, "lp_num":0, "lp_other":1, "domain_length":9,"domain":"gmx.com"}'
        parsed_json = json.loads(data_input)
        for key, value in parsed_json.items():
          parsed_json[key] = [value]
        model_input = T3S.preprocess_input_examples_arg_string('examples=['+json.dumps(parsed_json)+']')
        feature_chance = T3S.run_saved_model_with_feed_dict(dir_model, "serve", "predict", model_input,'./',True)
        json_result = {'feature_chance':np.float64(feature_chance) }  
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
        input_tensor_key_feed_dict: A dictionary maps input keys to numpy ndarrays.
        outdir: A directory to save the outputs to. If the directory doesn't exist,
            it will be created.
        overwrite_flag: A boolean flag to allow overwrite output file if file with
            the same name exists.
        tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the
            intermediate Tensor values and runtime GraphDefs while running the
            SavedModel.
      Raises:
        ValueError: When any of the input tensor keys is not valid.
        RuntimeError: An error when output file already exists and overwrite is not
        enabled.
      """
      result_string = ''
      # Get a list of output tensor names.
      meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,tag_set)

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
      """Create a serialized tf.example from feature dictionary."""
      example = example_pb2.Example()
      for feature_name, feature_list in example_dict.items():
        if not isinstance(feature_list, list):
          raise ValueError('feature value must be a list, but %s: "%s" is %s' %
                           (feature_name, feature_list, type(feature_list)))
        if isinstance(feature_list[0], float):
          example.features.feature[feature_name].float_list.value.extend(
              feature_list)
        elif isinstance(feature_list[0], str):
          for i in range(len(feature_list)):
            feature_list[i] = feature_list[i].encode()
          example.features.feature[feature_name].bytes_list.value.extend(
              feature_list)
        elif isinstance(feature_list[0], int):
          for i in range(len(feature_list)):
            feature_list[i] = float(feature_list[i])
          example.features.feature[feature_name].float_list.value.extend(
              feature_list)
        elif isinstance(feature_list[0], bytes):
          example.features.feature[feature_name].bytes_list.value.extend(
              feature_list)

        else:
          raise ValueError(
              'Type %s for value %s is not supported for tf.train.Feature.' %
              (type(feature_list[0]), feature_list[0]))
      return example.SerializeToString()


api.add_resource(T3S, '/<string:data_input>')

if __name__ == '__main__':
    app.run(debug=True)
