from os.path import join
import numpy as np
import nibabel as nib
import torch

from matplotlib import pyplot as plt
import os
import json
import glob

from SubSegmentation.lovasz import *
from SubSegmentation.processing_utils import *
import fetal_measure
import pandas as pd
import os
import json
import traceback

import argparse


def load_model():
    IMAGE_SIZE = (160, 160)
    MODEL_NAME = 'model-tfms'
    torch.cuda.set_device(0)

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    fm = fetal_measure.FetalMeasure(subseg_model='model-tfms')
    return fm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='fetal_measure',
        description='Fetal brain MRI linear measurements')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--infile', help='Input Nifti File to execute on')
    group.add_argument('-i', '--inputdir', help='Input directory')
    parser.add_argument('-o', '--outputdir', help='Output directory', required=True)
    args = parser.parse_args()

    if args.infile is not None:
        infiles = [os.path.abspath(args.infile), ]
    elif args.inputdir is not None:
        infiles = list(glob.glob(os.path.join(args.inputdir, "*.nii*")))
    else:
        raise ValueError('Missing input dir or files')

    fm = load_model()
    for a in infiles:
        try:
            fm.execute(a, out_dir=os.path.join(args.outputdir, os.path.basename(a)))
        except Exception:
            print ('Error in file:', a)
            print(traceback.format_exc())
            continue
