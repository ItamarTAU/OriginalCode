from os.path import join
import numpy as np
import nibabel as nib
import torch
from fastai.basic_train import load_learner
from matplotlib import pyplot as plt
import os
import json

import fetal_seg
from SubSegmentation.lovasz import *
from SubSegmentation.processing_utils import *
from SubSegmentation.brain_segmentation_model import BrainSegmentationModel
import slice_select
from scipy import ndimage

import CBD_BBD
import msl

import re

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

FD_RE = re.compile(
            "Pat(?P<patient_id>[\d]+)_Se(?P<series>[\d]+)_Res(?P<res_x>[\d.]+)_(?P<res_y>[\d.]+)_Spac(?P<res_z>[\d.]+).nii")


class FetalMeasure(object):
    def __init__(self, basedir="/home/netanell/work/FetalMeas2/Models/",
                 braigseg_roi_model="22-ROI",
                 braigseg_seg_model="24-seg",
                 subseg_model='model-tfms',
                 sliceselect_bbd_model="23_model_bbd",
                 sliceselect_tcd_model="24_model_tcd"):
        self.fetal_seg = fetal_seg.FetalSegmentation(
            os.path.join(basedir, "Brainseg", "22-ROI"), None,
            os.path.join(basedir, "Brainseg", "24-seg"), None)

        #self.subseg_learner = load_learner(os.path.join(basedir, "Subseg"), subseg_model)
        self.subseg = BrainSegmentationModel((160,160), os.path.join(basedir, "Subseg"), subseg_model)
        # self.sl_bbd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/FetalMeasurements/Models/Sliceselect/19_model_bbd",
        #     basemodel='ResNet50', cuda_id=0)
        # #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        # self.sl_tcd = slice_select.SliceSelect(
        #     model_file="/home/netanell/work/research/slice_select/models/7/epoch0029_02-02_1430_choose_acc0.9672.statedict.pkl", cuda_id=0)

        self.sl_bbd = slice_select.SliceSelect(
            model_file=os.path.join(basedir, "Sliceselect", "22_model_bbd"),
            basemodel='ResNet50', cuda_id=0)
        #model_file = "/home/netanell/work/research/slice_select/models/6/epoch0029_02-02_1249_choose_acc0.9678.statedict.pkl")
        self.sl_tcd = slice_select.SliceSelect(
            model_file=os.path.join(basedir, "Sliceselect", "25_model_tcd"),
            basemodel='ResNet50', cuda_id=0)
        print("Finished to load **************************")

    def _predict_nifti_subseg(self, img_data, filename):
        IMAGE_SIZE = (160, 160)
        images, min_ax, zeros, x_ax, y_ax = pre_processing(img_data, IMAGE_SIZE)
        segmentations_result = []
        zoomratio = min_ax/IMAGE_SIZE[0]
        for image in images:
            pred_class, pred_idx, outputs = self.subseg_learner.predict(image)
            segmentations_result.append(ndimage.zoom(pred_idx.data.squeeze(), zoomratio, order=0))



        seg_img = post_processing(segmentations_result, min_ax, zeros, x_ax, y_ax, filename)

    def execute(self, in_img_file, out_dir):
        # Full pipeline
        # in_img_file = "/media/df3-dafna/Netanell/DemoCode/Pat02_Se06_Res0.7422_0.7422_Spac5.nii"
        # out_dir = "/media/df3-dafna/Netanell/DemoCode/Outputs"

        elem_fname = os.path.basename(in_img_file)
        pat_id, ser_num, res_x, res_y, res_z = [float(x) for x in FD_RE.findall(elem_fname)[0]]

        metadata = {}
        metadata["InFile"] = in_img_file
        metadata["OutDir"] = out_dir
        metadata["Resolution"] = (res_x, res_y, res_z)
        metadata["SubjectID"] = pat_id
        metadata["Series"] = ser_num
		
		
        # Prep image
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.basename(in_img_file)
        elem_nii = nib.load(in_img_file)
        elem_nii_arr = elem_nii.get_fdata()
        input_axes = np.argsort(-np.array(elem_nii_arr.shape))
        nib_out = nib.Nifti1Image(elem_nii_arr.transpose(input_axes), affine=np.eye(4))
        reorint_image_niifile = os.path.join(out_dir, fn)
        nib.save(nib_out, reorint_image_niifile)
		
        # First stage - Segmentation
        self.fetal_seg.predict(reorint_image_niifile, out_dir)
        seg_file = os.path.join(out_dir, "prediction.nii.gz")
        subseg_file = os.path.join(out_dir, "subseg.nii.gz")
		

        # Second stage - Slice select
        sl_tcd_result = self.sl_tcd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file, )
        sl_tcd_slice = int(sl_tcd_result["prediction"].values[0])
        metadata["TCD_selection"] = sl_tcd_slice
        metadata["TCD_selectionValid"] = sl_tcd_result["isValid"].values[0]
        metadata["TCD_result"] = sl_tcd_result.prob_vec.tolist()

        sl_bbd_result = self.sl_bbd.execute(img_file=reorint_image_niifile,
                                            seg_file=seg_file, )
        sl_bbd_slice = int(sl_bbd_result["prediction"].values[0])
        metadata["BBD_selection"] = sl_bbd_slice
        metadata["BBD_result"] = sl_bbd_result.prob_vec.tolist()
        metadata["BBD_selectionValid"] = sl_bbd_result["isValid"].values[0]

        data_cropped, fullseg = self.sl_tcd.get_cropped_elem(img_file=reorint_image_niifile,
                                                             seg_file=seg_file, )

        data_cropped = data_cropped.transpose([1, 2, 0])
        fullseg = fullseg.transpose([1, 2, 0])

        nii_data_cropped = nib.Nifti1Image(data_cropped, affine=np.eye(4))
        nib.save(nii_data_cropped, os.path.join(out_dir, "cropped.nii.gz"))

        nii_seg_cropped = nib.Nifti1Image(fullseg.astype(float), affine=np.eye(4))
        nib.save(nii_seg_cropped, os.path.join(out_dir, "seg.cropped.nii.gz"))

        # Third stage - Sub segmentaion

        #self._predict_nifti_subseg(data_cropped.copy(), subseg_file)
        self.subseg.predict_nifti(data_cropped.copy(), subseg_file, tta=True)
        subseg = nib.load(subseg_file).get_fdata()
        # Fourth stage - MSL
        msl_p_planes = msl.find_planes(data_cropped, subseg)
        msl_p_points = msl.findMSLforAllPlanes(data_cropped, subseg, msl_p_planes)
        metadata["msl_planes"] = msl_p_planes
        metadata["msl_points"] = msl_p_points

        # Fifth - Measuring

        # BBD + CBD
        CBD_min_th = (subseg.shape[0] / 4)
        p_u, p_d, _ = msl_p_points[sl_bbd_slice]
        CBD_measure, CBD_left, CBD_right = CBD_BBD.CBD_points(subseg[:, :, sl_bbd_slice] > 0,
                                                                   p_u.astype(int),
                                                                   p_d.astype(int), res_x, res_y,
                                                                  CBD_min_th)
        metadata["cbd_points"] = (CBD_left, CBD_right)
        metadata["cbd_measure_mm"] = CBD_measure
        BBD_measure, BBD_left, BBD_right, BBD_valid = CBD_BBD.BBD_points(data_cropped[:, :, sl_bbd_slice], CBD_left, CBD_right,
                                                             p_u.astype(int), p_d.astype(int),
                                                  res_x, res_y)
        metadata["bbd_points"] = (BBD_left, BBD_right)
        metadata["bbd_measure_mm"] = BBD_measure
        metadata["bbd_valid"] = BBD_valid

        print(np.linalg.norm(CBD_left - CBD_right) * res_x, np.linalg.norm(BBD_left - BBD_right) * res_x)
        plt.figure()
        plt.imshow(data_cropped[:, :, sl_bbd_slice])
        cbd = np.stack([CBD_left, CBD_right]).T
        plt.plot(cbd[1, :], cbd[0, :], 'r')
        plt.savefig(os.path.join(out_dir, "cbd.png"))
        plt.figure()
        plt.imshow(data_cropped[:, :, sl_bbd_slice])
        bbd = np.stack([BBD_left, BBD_right]).T
        plt.plot(bbd[1, :], bbd[0, :], 'b-')
        plt.savefig(os.path.join(out_dir, "bbd.png"))

        # TCD
        plt.figure()
        p_u, p_d, _ = msl_p_points[sl_tcd_slice]
        TCD_measure, TCD_left, TCD_right, TCD_valid = CBD_BBD.TCD_points(subseg[:, :, sl_tcd_slice] == 2., p_u.astype(int), p_d.astype(int),
                                                 res_x, res_y)
        metadata["tcd_valid"] = TCD_valid
        metadata["tcd_points"] = (TCD_left, TCD_right)
        metadata["tcd_measure_mm"] = TCD_measure
        plt.imshow(data_cropped[:, :, sl_tcd_slice])
        if TCD_measure is not None:
            tcd = np.stack([TCD_left, TCD_right]).T
            plt.plot(tcd[1, :], tcd[0, :], 'k-')
            plt.savefig(os.path.join(out_dir, "tcd.png"))

        # Brain Volume Calc

        metadata["brain_vol_voxels"] = float(np.sum(fullseg > .5))
        metadata["brain_vol_mm3"] = float(np.sum(fullseg > .5) * res_x * res_y * res_z)

        # Dump metadata
        with open(os.path.join(out_dir, 'data.json'), 'w') as fp:
            json.dump(metadata, fp, cls=NumpyEncoder)

        return metadata
