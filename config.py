
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--alpha', default=0.02)
parser.add_argument('--beta', default=0.01)
parser.add_argument('--max_tau', default=5)
parser.add_argument('--save_path', default='./result')

args = parser.parse_args()
