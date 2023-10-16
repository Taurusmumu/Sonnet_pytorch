import math
import cv2
import numpy as np
import scipy.io as sio


def get_tk_label():
    lb_list = []
    label_num = 8
    for j in range(label_num + 1):
        t_k = math.exp(math.log(2) - math.log(2) * (j / label_num)) - 1
        lb_list.append(t_k)
    return lb_list


def tk_labelize(label_list, input_np):
    output_np = np.zeros_like(input_np)
    for i in range(input_np.shape[0]):
        value = input_np[i]
        if value == 1:
            output_np[i] = 1
            continue
        for idx in range(len(label_list) - 1):
            if value < label_list[idx] and value >= label_list[idx + 1]:
                output_np[i] = idx + 1
    return output_np


tk_label_list = get_tk_label()


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.

    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane,
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann

class __MoNuSAC(__AbstractDataset):

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_msk(self, path):
        return cv2.imread(path)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        original_maps = sio.loadmat(path)
        inst_map = original_maps['inst_map'].astype(int)
        inst_centroid = original_maps['inst_centroid']
        ordinal_map = np.full((inst_map.shape[0], inst_map.shape[1]), 0, dtype='float32')
        for uid in np.unique(inst_map):
            if uid == 0:
                continue
            cent = inst_centroid[uid - 1]
            mask = (inst_map == uid).astype(int)
            coordinate_np = np.array([np.where(mask == True)[0], np.where(mask == True)[1]])
            coordinate_np = np.sqrt(
                np.power(coordinate_np[0, :] - cent[1], 2) + np.power(coordinate_np[1, :] - cent[0], 2))
            coordinate_np /= max(coordinate_np)
            ordinal_map[np.where(mask > 0)] = tk_labelize(tk_label_list, coordinate_np)

        if with_type:
            ann_type = original_maps["type_map"]
            ann = np.dstack([original_maps["inst_map"], ann_type, ordinal_map])
            ann = ann.astype("int32")
        else:
            ann = np.dstack([original_maps["inst_map"], ordinal_map])
            ann = ann.astype("int32")

        return ann

####
class __GLySAC(__AbstractDataset):

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_msk(self, path):
        return cv2.imread(path)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        original_maps = sio.loadmat(path)
        inst_map = original_maps['inst_map'].astype(int)
        inst_centroid = original_maps['inst_centroid']
        img_size = inst_map.shape[0]
        ordinal_map = np.full((img_size, img_size), 0, dtype='float32')
        for uid in np.unique(inst_map):
            if uid == 0:
                continue
            cent = inst_centroid[uid - 1]
            mask = (inst_map == uid).astype(int)
            coordinate_np = np.array([np.where(mask == True)[0], np.where(mask == True)[1]])
            coordinate_np = np.sqrt(np.power(coordinate_np[0, :] - cent[1], 2) + np.power(coordinate_np[1, :] - cent[0], 2))
            coordinate_np /= max(coordinate_np)
            ordinal_map[np.where(mask > 0)] = tk_labelize(tk_label_list, coordinate_np)

        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 1) | (ann_type == 2) | (ann_type == 9) | (ann_type == 10)] = 1
            ann_type[(ann_type == 4) | (ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 2
            ann_type[(ann_type == 3) | (ann_type == 8)] = 3
            ann = np.dstack([original_maps["inst_map"], ann_type, ordinal_map])
            ann = ann.astype("int32")
        else:
            ann = np.dstack([original_maps["inst_map"], ordinal_map])
            ann = ann.astype("int32")

        return ann


####
class __CoNSeP(__AbstractDataset):
    """
    Defines the CoNSeP dataset as originally introduced in:
    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak,
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_msk(self, path):
        return cv2.imread(path)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        original_maps = sio.loadmat(path)
        inst_map = original_maps['inst_map'].astype(int)
        inst_centroid = original_maps['inst_centroid']
        img_size = inst_map.shape[0]
        ordinal_map = np.full((img_size, img_size), 0, dtype='float32')
        for uid in np.unique(inst_map):
            if uid == 0:
                continue
            cent = inst_centroid[uid - 1]
            mask = (inst_map == uid).astype(int)
            coordinate_np = np.array([np.where(mask == True)[0], np.where(mask == True)[1]])
            coordinate_np = np.sqrt(
                np.power(coordinate_np[0, :] - cent[1], 2) + np.power(coordinate_np[1, :] - cent[0], 2))
            coordinate_np /= max(coordinate_np)
            ordinal_map[np.where(mask > 0)] = tk_labelize(tk_label_list, coordinate_np)

        if with_type:
            ann_type = original_maps["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([original_maps["inst_map"], ann_type, ordinal_map])
            ann = ann.astype("int32")
        else:
            ann = np.dstack([original_maps["inst_map"], ordinal_map])
            ann = ann.astype("int32")

        return ann


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        # "kumar": lambda: __Kumar(),
        "monusac": lambda: __MoNuSAC(),
        "glysac": lambda: __GLySAC(),
        "consep": lambda: __CoNSeP(),
    }
    if name.lower() in name_dict:
        return name_dict[name.lower()]()
    else:
        assert False, "Unknown dataset `%s`" % name
