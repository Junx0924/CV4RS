import argparse
from fsplit.filesplit import Filesplit

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True, type = str)

args = vars(parser.parse_args())
path = args.pop('path')

fs = Filesplit()

MAX_SIZE = int(99*1e6) # 99 MB
fs.split(path, MAX_SIZE)
