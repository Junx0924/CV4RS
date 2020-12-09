from .base import *
from pathlib import Path
import pandas as pd


class MLRSNet(BaseDataset):
    def __init__(self, root, classes, transform = None):
        BaseDataset.__init__(self, root, classes, transform)

        base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        base_dir = str(base_dir.parent.parent.parent)

        # load train/val split
        if (classes == "train") | (classes == "init"):
            patches = pd.read_csv(base_dir + "/data_splits/MLRSNet/train.csv", sep=",")
            # with open(base_dir + "/data_splits/MLRSNet/train.csv", "r") as f:
            #    patches = f.readlines()
            # patches = [x.replace("\n", "") for x in patches]
        elif classes == "eval":
            with open(base_dir + "/data_splits/MLRSNet/val.csv", "r") as f:
                patches = f.readlines()
            patches = [x.replace("\n", "") for x in patches]
        else:
            print('Unknown value for classes selected')
            raise Exception

        i = 0
        for count in range(len(patches)):
            row = patches[count]
            print(classes)
            print(row)
            raise Exception

    def nb_classes(self):
        assert len(set(self.ys)) == len(set(self.classes))
        return len(self.classes)
