from .lovasz import *
from .processing_utils import *
from fastai.basic_train import load_learner
from os.path import join

IMAGE_SIZE = (160, 160)

MODEL_NAME = 'model-tfms'
learn = load_learner(join('/media/df3-dafna/Ori/models'), MODEL_NAME)


def predict_nifti(img_data, dest_filename):
    images, min_ax, zeros, x_ax, y_ax = pre_processing(img_data, IMAGE_SIZE)
    segmentations_result = []

    for image in images:
        pred_class, pred_idx, outputs = learn.predict(image)
        segmentations_result.append(pred_idx.data.squeeze())

    post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, dest_filename)