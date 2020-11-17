"""
Given a TensorFlow Savedmodel (1.x) and desired output nodes, create a TensorFlow.js model.
"""
import os
import shutil
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflowjs.converters.tf_saved_model_conversion_v2 import convert_tf_frozen_model


def freeze_savedmodel(savedmodel_dir, output_nodes, frozen_file, tags=None):
	"""
	Convert a SavedModel to a given frozen graph .pb file
	"""
	if tags is None:
		tags = [tf.saved_model.SERVING]
	# our output node names must exist in the graph, but also not include the : selector that tensorflow has by default
	output_node_names = ",".join([out.split(":")[0] for out in output_nodes])
	freeze_graph(
		input_graph=None,
		input_saver=None,
		input_binary=False,
		input_checkpoint=None,
		output_node_names=output_node_names,
		restore_op_name=None,
		filename_tensor_name=None,
		output_graph=frozen_file,
		clear_devices=True,
		initializer_nodes="",
		variable_names_whitelist="",
		variable_names_blacklist="",
		input_meta_graph=None,
		input_saved_model_dir=savedmodel_dir,
		saved_model_tags=",".join(tags),
	)


def convert_savedmodel(savedmodel_dir, output_nodes, output_dir, tags=None):
	"""
	Given a saved model, desired output nodes, and tags, convert to tfjs
	"""
	# convert to frozen graph file
	frozen_filename = os.path.join(savedmodel_dir, 'tmp_frozengraph.pb')
	try:
		freeze_savedmodel(
			savedmodel_dir=savedmodel_dir, output_nodes=output_nodes, frozen_file=frozen_filename, tags=tags
		)
	except Exception as e:
		print(f"Error freezing graph: {e}")
		if os.path.isfile(frozen_filename):
			os.remove(frozen_filename)
		raise

	# now convert the frozen graph to the tfjs model
	try:
		convert_tf_frozen_model(
			frozen_model_path=frozen_filename,
			output_node_names=",".join(output_nodes),
			output_dir=output_dir
		)
	except Exception as e:
		print(f"Error converting frozen graph: {e}")
		if os.path.exists(output_dir):
			shutil.rmtree(output_dir)
		raise
	finally:
		if os.path.exists(frozen_filename):
			os.remove(frozen_filename)
