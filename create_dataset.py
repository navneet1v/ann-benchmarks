import argparse

from ann_benchmarks.datasets import DATASETS, get_dataset_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--generate_ground_truth", choices=['yes', 'no'], required=False, default='yes')
    args = parser.parse_args()
    fn = get_dataset_fn(args.dataset)
    DATASETS[args.dataset](fn, args.generate_ground_truth)
