import glob
import inspect
import logging
import os
import shutil
import random
import torch
import cv2
import numpy as np
from scipy import ndimage
import re

__ALL__ = [ 'colored', 'cprint' ]

VERSION = (1, 1, 0)

ATTRIBUTES = dict(
        list(zip([
            'bold',
            'dark',
            '',
            'underline',
            'blink',
            '',
            'reverse',
            'concealed'
            ],
            list(range(1, 9))
            ))
        )
del ATTRIBUTES['']
HIGHLIGHTS = dict(
        list(zip([
            'on_grey',
            'on_red',
            'on_green',
            'on_yellow',
            'on_blue',
            'on_magenta',
            'on_cyan',
            'on_white'
            ],
            list(range(40, 48))
            ))
        )
COLORS = dict(
        list(zip([
            'grey',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
            ],
            list(range(30, 38))
            ))
        )
RESET = '\033[0m'

def check_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        print(f'File exists: {path}')


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


####
def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


####
def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


####
def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.

    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


####
def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


####
def log_debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


####
def log_info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.

    Args:
        seed: seed to check

    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ia.random.seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return


def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        fmt_str = '\033[%dm%s'
        if color is not None:
            text = fmt_str % (COLORS[color], text)

        if on_color is not None:
            text = fmt_str % (HIGHLIGHTS[on_color], text)

        if attrs is not None:
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)

        text += RESET
    return text

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict