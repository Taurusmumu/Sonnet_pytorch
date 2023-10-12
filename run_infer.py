"""run_infer.py

Usage:
  run_infer.py [options] [--help] [command] [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 3]
  --nr_types=<n>              Number of nuclei types to predict. Set 4 if glysac or 5 if consep [default: 5]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: ./type_info.json]

  --model_path=<path>         Path to saved checkpoint. [default: ./logs/consep_88888888/consep_seed/01/net_epoch=45.tar]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   not support yet

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile [--input_dir=<path>] [--output_dir=<path>] [--draw_dot=<n>] [--save_qupath=<n>] [--save_raw_map=<n>] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory. [default: ./dataset/CoNSeP/Test/Images]
   --output_dir=<path>    Path to output directory.. [default: ./test_output/consep_88888888/consep_seed]

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot=<n>             To draw nuclei centroid on overlay. [default: True]
   --save_qupath=<n>          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map=<n>         To save raw prediction or not. [default: False]
"""

import torch
import logging
import os
from misc.utils import log_info
from docopt import docopt

#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    sub_cli_dict = {'tile' : tile_cli}
    args = docopt(__doc__, help=False, options_first=True, version='Sonnet Pytorch Inference v1.0')
    sub_cmd = 'tile'
    sub_cmd_args = args.pop('<args>')

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
    
    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info('Detect #GPUS: %d' % nr_gpus)

    args = {k.replace('--', '') : v for k, v in args.items()}
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}
    if args['model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                "num_classes": 1024,
                "nt_class_num": nr_types,
                "nf_class_num": 2,
                "no_class_num": 16,
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' else args['type_info_path'],
    }

    # ***
    run_args = {
        'batch_size' : int(args['batch_size']) * nr_gpus,

        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }
    run_args['patch_input_shape'] = 270
    run_args['patch_output_shape'] = 76

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],

            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : sub_args['draw_dot'],
            'save_qupath' : sub_args['save_qupath'],
            'save_raw_map': sub_args['save_raw_map'],
        })
    # ***
    
    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        assert "Can not handle other type yet"
