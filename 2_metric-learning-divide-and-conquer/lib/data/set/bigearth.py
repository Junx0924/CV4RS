from .base import *
import json
from pathlib import Path


class BigEarth(BaseDataset):
    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)

        base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        base_dir = str(base_dir.parent.parent.parent)

        with open(base_dir + "/label_indicies/BigEarthNet/label_indices.json", "r") as f:
            label_indices = json.load(f)['original_labels']

        # make a train/test split
        if (classes == "train") | (classes == "init"):
            with open(base_dir + "/data_splits/BigEarthNet/train.csv", "r") as f:
                patches = f.readlines()
            patches = [x.replace("\n", "") for x in patches]
        elif classes == "eval":
            with open(base_dir + "/data_splits/BigEarthNet/val.csv", "r") as f:
                patches = f.readlines()
            patches = [x.replace("\n", "") for x in patches]
        else:
            print('Unknown value for classes selected')
            raise Exception

        i = 0
        for patch_folder in patches:
            if not os.path.isdir(root + "/" + patch_folder):
                continue
            patch_folder_content = os.listdir(root + "/" + patch_folder)
            assert (len(patch_folder_content) == 13), "There are not 13 elements in the folder " + patch_folder
            patch_labels_name = [filename for filename in patch_folder_content if filename.endswith(".json")][0]
            with open(root + "/" + patch_folder + "/" + patch_labels_name, 'r') as file:
                patch_labels = json.load(file)['labels']

            # add image to data set for each label
            for label in patch_labels:
                index_of_label = int(label_indices[label])
                self.im_paths += [root+"/"+patch_folder+"/"+patch_folder]
                self.ys += [index_of_label]
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
