import os
import numpy as np
from scipy import ndimage

import sys

sys.path.append("./fetal_segmentation")

from fetal_segmentation.evaluation.predict_nifti_dir import get_params, preproc_and_norm, \
    get_prediction, secondary_prediction
from fetal_segmentation.training.train_functions.training import load_old_model, get_last_model_path
from fetal_segmentation.utils.read_write_data import save_nifti, read_img
from fetal_segmentation.data_curation.helper_functions import move_smallest_axis_to_z, \
    swap_to_original_axis
from fetal_segmentation.evaluation.eval_utils.postprocess import postprocess_prediction



class FetalSegmentation(object):
    def __init__(self, config_roi_dir, model_roi,
                 config_secondnet_dir, model_net,
                 ):

        self._config, self._norm_params, self._model_path = get_params(config_roi_dir)

        # LOad second network if possible
        if config_secondnet_dir is not None:
            self._config2, self._norm_params2, self._model2_path = get_params(config_secondnet_dir)
        else:
            self._config2, self._norm_params2, self._model2_path = None, None, None

        print('First:' + self._model_path)
        self._model = load_old_model(get_last_model_path(self._model_path), config=self._config)

        if (self._model2_path is not None):
            print('Second:' + self._model2_path)
            self._model2 = load_old_model(get_last_model_path(self._model2_path), config=self._config2)
        else:
            self._model2 = None

    def predict(self, in_file, output_path,
                overlap_factor=0.7,
                z_scale=None, xy_scale=None):
        # def main(input_path, output_path, has_gt, scan_id, overlap_factor,
        #      config, model, preprocess_method=None, norm_params=None, augment=None, num_augment=0,
        #      config2=None, model2=None, preprocess_method2=None, norm_params2=None, augment2=None, num_augment2=0,
        #      z_scale=None, xy_scale=None, return_all_preds=False):

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print('Loading nifti from {}...'.format(in_file))
        nifti_data = read_img(in_file).get_fdata()
        print('Predicting mask...')
        save_nifti(nifti_data, os.path.join(output_path, 'data.nii.gz'))
        nifti_data, swap_axis = move_smallest_axis_to_z(nifti_data)
        data_size = nifti_data.shape
        data = nifti_data.astype(np.float).squeeze()
        print('original_shape: ' + str(data.shape))

        if (z_scale is None):
            z_scale = 1.0
        if (xy_scale is None):
            xy_scale = 1.0
        if z_scale != 1.0 or xy_scale != 1.0:
            data = ndimage.zoom(data, [xy_scale, xy_scale, z_scale])

        data = preproc_and_norm(data, preprocess_method="window_1_99", norm_params=self._norm_params,
                                scale=self._config.get('scale_data', None),
                                preproc=self._config.get('preproc', None))

        print('Shape: ' + str(data.shape))
        prediction = get_prediction(data=data, model=self._model, augment=None,
                                    num_augments=1, return_all_preds=None,
                                    overlap_factor=overlap_factor, config=self._config)

        # revert to original size
        if self._config.get('scale_data', None) is not None:
            prediction = \
                ndimage.zoom(prediction.squeeze(), np.divide([1, 1, 1], self._config.get('scale_data', None)), order=0)[
                    ..., np.newaxis]

        if z_scale != 1.0 or xy_scale != 1.0:
            prediction = prediction.squeeze()
            prediction = ndimage.zoom(prediction,
                                      [data_size[0] / prediction.shape[0], data_size[1] / prediction.shape[1],
                                       data_size[2] / prediction.shape[2]], order=1)[..., np.newaxis]

        prediction = prediction.squeeze()

        mask = postprocess_prediction(prediction, threshold=0.5)

        if self._config2 is not None:
            swapped_mask = swap_to_original_axis(swap_axis, mask)
            save_nifti(np.int16(swapped_mask), os.path.join(output_path, 'prediction_all.nii.gz'))
            prediction = secondary_prediction(mask, vol=nifti_data.astype(np.float),
                                              config2=self._config2, model2=self._model2,
                                              preprocess_method2="window_1_99", norm_params2=self._norm_params2,
                                              overlap_factor=overlap_factor, augment2=None,
                                              num_augment=1,
                                              return_all_preds=None)

            prediction_binarized = postprocess_prediction(prediction, threshold=0.5)
            prediction_binarized = swap_to_original_axis(swap_axis, prediction_binarized)
            save_nifti(np.int16(prediction_binarized), os.path.join(output_path, 'prediction.nii.gz'))

        else:  # if there is no secondary prediction, save the first network prediction or predictions as the final ones
            mask = swap_to_original_axis(swap_axis, mask)
            save_nifti(np.int16(mask), os.path.join(output_path, 'prediction.nii.gz'))
        print('Saving to {}'.format(output_path))
        print('Finished.')
