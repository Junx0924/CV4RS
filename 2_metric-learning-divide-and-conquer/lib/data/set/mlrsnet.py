from .base import *
from pathlib import Path
import pandas as pd
import json


class MLRSNet(BaseDataset):
    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)

        base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        base_dir = str(base_dir.parent.parent.parent)

        with open(base_dir + "/label_indicies/MLRSNet/label_name.json", "r") as f:
            label_name = json.load(f)
        label_name = {y: x for x, y in label_name.items()}

        # load train/val split
        if (classes == "train") | (classes == "init"):
            patches = pd.read_csv(base_dir + "/data_splits/MLRSNet/train.csv", sep=",",
                                  names=["path_img"]+["l%i" % x for x in range(60)])
        elif classes == "eval":
            patches = pd.read_csv(base_dir + "/data_splits/MLRSNet/val.csv", sep=",",
                                  names=["path_img"] + ["l%i" % x for x in range(60)])
        else:
            print('Unknown value for classes selected')
            raise Exception

        i = 0
        for count in range(len(patches)):
            row = patches.loc[0, :]
            folder_name = row.path_img.split("/")[2]
            label_path = root + "/labels/" + folder_name + ".csv"
            img_file_name = row.path_img.split("/")[-1]

            labels_file = pd.read_csv(label_path, sep=",")
            labels_of_img = labels_file[labels_file.image.str.startswith(img_file_name)]

            l = len(labels_of_img)
            if l != 1:
                if l > 1:
                    print("More than 1 match")
                if l < 1:
                    print("No match")
                raise Exception
            else:
                # iterate over multi-labels
                for label in labels_of_img.columns[1:]:
                    is_label_present = labels_of_img[label].values[0]
                    if is_label_present == 1:
                        # the label 'label' is present for the given image 'img_file_name'
                        label_id = int(label_name[label])
                        # print(label + " (%i): %i" % (label_id, is_label_present))
                        self.im_paths += [root + row.path_img]
                        self.ys += [label_id]
                        self.I += [i]
                        i += 1

        if (classes == "train") | (classes == "init"):
            # print("Train", len(set(self.ys)))
            self.classes = range(0, len(set(self.ys)))
        elif classes == "eval":
            # print("Eval", len(set(self.ys)))
            self.classes = range(0, len(set(self.ys)))

    def nb_classes(self):
        assert len(set(self.ys)) == len(set(self.classes))
        return len(self.classes)
