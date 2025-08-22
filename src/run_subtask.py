
import argparse
from pathlib import Path
from generate_data import generate_data, ALL_TASK_CHOICES
from run_eval import run_eval

def main():

	parser = argparse.ArgumentParser(description="LLM Mapping Dataset Runner")
	parser.add_argument('--model', type=str, default='flash', help='Model name for evaluation')
	parser.add_argument('--size', type=int, default=100, help='Number of examples to generate')
	parser.add_argument('--tasks', type=str, nargs='+', choices=ALL_TASK_CHOICES, default=ALL_TASK_CHOICES, help='Task types (space separated)')
	parser.add_argument('--output', type=str, default='./data/examples.jsonl', help='Output path for generated data')
	parser.add_argument('--eval', action='store_true', default=True, help='Run evaluation after data generation (default: True)')

	args = parser.parse_args()

	# Generate data
	generate_data(args.tasks, args.size, args.output)

	# Run evaluation if requested
	if args.eval:
		run_eval(args.model, args.output)

if __name__ == "__main__":
	main()
