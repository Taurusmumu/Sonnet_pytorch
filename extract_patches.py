import math
import scipy.io as sio
import re
import glob
import os
import tqdm
import pathlib
import numpy as np
from dataset import get_dataset
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
from misc.utils import sorted_alphanumeric


if __name__ == "__main__":
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
    dataset_name = "CoNSeP"
    # dataset_name = "GLySAC"

    save_root = "dataset/training_data"
    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".png" if dataset_name.lower() == "consep" else ".tif", f"dataset/{dataset_name}/Train/Images/"),
            "ann": (".mat", f"dataset/{dataset_name}/Train/Labels/"),
        },
        "valid": {
            "img": (".png" if dataset_name.lower() == "consep" else ".tif", f"dataset/{dataset_name}/Test/Images/"),
            "ann": (".mat", f"dataset/{dataset_name}/Test/Labels/"),
        },
    }
    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]
        if "train_mix" in split_name:
            msk_ext, msk_dir = split_desc["msk"]

        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name.lower(),
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )
            if "train_mix" in split_name:
                msk = parser.load_msk(
                    "%s/%s%s" % (msk_dir, base_name + "_mask_nuclei2", msk_ext)
                )
            # *
            img = np.concatenate([img, ann], axis=-1)
            if "train_mix" in split_name:
                img = np.concatenate([img, msk], axis=-1)

            sub_patches = xtractor.extract(img, extract_type)
            new_sub_patches = []
            if "train_mix" in split_name:
                for patch in sub_patches:
                    msk = patch[:, :, -1]
                    if 0 in msk:
                        new_sub_patches.append(patch[:, :, :-3])
                sub_patches = new_sub_patches

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *
            pbarx.update()
        pbarx.close()
