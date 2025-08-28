import os
import argparse
from datetime import datetime
from generate_data import generate_data, ALL_TASK_CHOICES
from run_eval import run_eval

def main():

	parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
	parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
	parser.add_argument('--size', type=int, default=100, help='Number of examples to generate')
	parser.add_argument('--task', type=str, nargs='+', choices=ALL_TASK_CHOICES, default=ALL_TASK_CHOICES, help='Task types (space separated)')
	parser.add_argument('--output', type=str, default='./data/examples.jsonl', help='Output path for generated data')
	parser.add_argument('--save_outputs', action='store_true', default=False, help='Save results and log files in runs directory (default: False)')
	parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
	parser.add_argument('--eval', action='store_true', default=True, help='Run evaluation after data generation (default: True)')

	args = parser.parse_args()

	# Inject timestamp into output path
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	base, ext = os.path.splitext(args.output)
	output_with_ts = f"{base}_{timestamp}{ext}"

	# Generate data
	generate_data(args.task, args.size, output_with_ts)

	# Run evaluation if requested
	if args.eval:
		run_eval(args.model, output_with_ts, save_outputs=args.save_outputs, verbose=args.verbose)

if __name__ == "__main__":
	main()
