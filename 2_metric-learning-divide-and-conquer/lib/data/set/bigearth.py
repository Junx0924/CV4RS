from .base import *
import json
import random


class BigEarth(BaseDataset):
    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)
        with open(root+"/label_indices.json", "r") as f:
            label_indices = json.load(f)['original_labels']

        all_patches = [name for name in os.listdir(root) if os.path.isdir(root + "/" + name)]
        all_patches = all_patches[:500]
        all_patches_half_len = len(all_patches) // 2
        random.Random(5).shuffle(all_patches)

        # make a train/test split
        if (classes == "train") | (classes == "init"):
            patches = all_patches[:all_patches_half_len]
        elif classes == "eval":
            patches = all_patches[all_patches_half_len:]
        else:
            print('Unknown value for classes selected')
            raise Exception

        i = 0
        for patch_folder in patches:
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
            self.classes = range(0, len(set(self.ys)))
        elif classes == "eval":
            self.classes = range(0, len(set(self.ys)))

    def nb_classes(self):
        assert len(set(self.ys)) == len(set(self.classes))
        return len(self.classes)
