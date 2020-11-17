"""
Given a Lobe TensorFlow export, create a TensorFlow.js model
"""
import argparse
import os
import json
from src.savedmodel_to_tfjs import convert_savedmodel


def convert_lobe_tfjs(save_dir, output_dir=None):
	"""
	Given the path to a Lobe TensorFlow export directory, convert the model to a TensorFlow.js model.
	"""
	signature_filename = 'signature.json'
	save_dir = os.path.realpath(save_dir)
	if output_dir is None:
		output_dir = os.path.join(save_dir, 'tensorflowjs_model')
	output_dir = os.path.realpath(output_dir)
	print(f'Converting saved model: {save_dir} to tfjs: {output_dir}')
	# load our signature json file, this shows us the model inputs and outputs
	with open(os.path.join(save_dir, signature_filename), 'r') as f:
		signature = json.load(f)
	tags = signature.get('tags')
	# prune our gather operation (because it uses a string input to get the labels, which tfjs does not currently support
	prune_signature(signature=signature)
	output_node_names = [out.get('name') for out in signature.get('outputs').values()]
	# convert!
	convert_savedmodel(savedmodel_dir=save_dir, output_nodes=output_node_names, tags=tags, output_dir=output_dir)
	# modify and save our signature.json to the output dir
	signature['filename'] = 'model.json'
	signature['format'] = 'tf_js'
	with open(os.path.join(output_dir, signature_filename), 'w') as f:
		json.dump(signature, f)
	print(f'Saved to {output_dir}')


def prune_signature(signature):
	"""
	Given a Lobe signature dictionary, prune in-place any outputs that are incompatible with tfjs
	"""
	outputs = signature.get('outputs', {})
	keys_to_delete = []
	for out_key, out in outputs.items():
		if 'Gather' in out.get('name'):
			keys_to_delete.append(out_key)
	print(f'Pruning outputs {keys_to_delete}')
	for out_key in keys_to_delete:
		del outputs[out_key]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert a Lobe TensorFlow model to a TensorFlow.js model.')
	parser.add_argument('save_dir', help='Path to your exported TensorFlow model directory.')
	parser.add_argument(
		'--output_dir', help='Path to your desired TensorFlow.js model directory.', default=None, required=False
	)
	args = parser.parse_args()
	save_dir = args.save_dir.replace('"', '')
	out_dir = args.output_dir
	if out_dir:
		out_dir = out_dir.replace('"', '')
	convert_lobe_tfjs(save_dir=save_dir, output_dir=out_dir)
