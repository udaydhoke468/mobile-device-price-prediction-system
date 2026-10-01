"""Train and save the classifier: python train_model.py data.csv [model.joblib]"""
import argparse
from model import train_from_csv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", help="Labeled CSV with model features and price_range (0–3)")
parser.add_argument("--output", default="model.joblib", help="Output model path")
args = parser.parse_args()
train_from_csv(args.csv, args.output)
print(f"Saved trained model to {args.output}")
