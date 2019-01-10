from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", default="aerts")
parser.add_argument("--seed", default=100, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--dropout-conv", default=0.25, type=float)
parser.add_argument("--dropout-fc", default=0.5, type=float)
