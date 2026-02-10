# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import json
import torch
import tempfile
from os import path as osp
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.common.data_classes import EvalBoxes
import mmcv
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS 
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
# from mmdet3d.datasets import NuScenesMonoDataset, NuScenesDataset
import os
import copy
from mmdet3d.structures.bbox_3d import CameraInstance3DBoxes, LiDARInstance3DBoxes, get_box_type, Box3DMode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
from typing import Callable, List, Union
from mmengine.structures import InstanceData
from torchvision.ops import box_iou
@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenesMono Dataset.
    This dataset add camera intrinsics and extrinsics and 2d bbox to the results.
    """
    def __init__(self, ann_file_2d, mini=False, load_separate=False, **kwargs):
        self.load_separate = load_separate
        self.ann_file_2d = ann_file_2d
        self.mini = mini
        super(CustomNuScenesDataset, self).__init__(**kwargs)
        self.load_annotations_2d(ann_file_2d)
        self.with_velocity  = False
    
    def __len__(self):
        return super(CustomNuScenesDataset, self).__len__()
    
    def filter_data(self):
        # get 10 random samples for debugging shuffled
        
        if True:
            token = 'a7831d4d1db54053a501d0418545fee2'
            new_data_list = []
            for info in self.data_list:
                if info['token'] == token:
                    new_data_list.append(info)
            self.data_list = new_data_list*10
            return self.data_list
        elif self.mini:
            # num_samples = 10
            original_data_list = self.data_list
            # np.random.seed(42)
            # shuffled_indices = np.random.permutation(len(original_data_list))
            shuffled_indices = np.array([17050, 31682, 32461, 33640, 4387, 5797, 32258, 3432, 28618, 7005, 
                                         14753, 30383, 23006, 27344, 12790, 14268, 13735, 7764, 10023, 9516, 
                                         28943, 16923, 5111, 31179, 30704, 30800, 3105, 18525, 12219, 2313, 
                                         19388, 3409, 21782, 15765, 27567, 7099, 31085, 9075, 18900, 10351, 
                                         2412, 13900, 33406, 19785, 11910, 19081, 22952, 13624, 7708, 33381, 
                                         7712, 25565, 21508, 25893, 31461, 8590, 12195, 8359, 27535, 712, 2023, 
                                         33961, 30484, 26053, 1610, 27296, 29765, 22133, 14033, 34127, 19799, 29803, 
                                         32179, 25103, 28952, 346, 18481, 15190, 19390, 15636, 25441, 23641, 19274, 
                                         25380, 11526, 33205, 1526, 14254, 29326, 16198, 7901, 2723, 29805, 16718, 24960, 
                                         7197, 27121, 18754, 33004, 29935])
            # np.array([8211, 1253, 6224, 6493, 32746, 31644, 10448, 21114, 35166, 22658])
            
            # self.data_list = [original_data_list[i] for i in shuffled_indices[:num_samples]]
            self.data_list = [original_data_list[i] for i in shuffled_indices]
            # save the loaded data_list tokens for reference
            self.loaded_tokens = []
            for info in self.data_list:
                self.loaded_tokens.append(info['token'])
            # print all loaded tokens
            print("Loaded mini dataset with tokens:")
            for token in self.loaded_tokens:
                print(token)
                
            # save tokens to a txt file
            with open("loaded_mini_dataset_tokens.txt", "w") as f:
                for token in self.loaded_tokens:
                    f.write(token + "\n")
                
        return self.data_list
    
    # def filter_data(self):
    #     original_data_list = self.data_list
    #         # np.random.seed(42)
    #         # shuffled_indices = np.random.permutation(len(original_data_list))
    #     token = "05e312eae68c454c86cea669ee26ced1"
    #     new_data_list = []
    #     for info in original_data_list:
    #         if info['token'] == token:
    #             new_data_list.append(info)
    #     self.data_list = new_data_list
        
    #     return self.data_list
    
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos_ori = data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        if self.load_separate:
            data_infos_path = []
            out_dir = self.ann_file.split('.')[0]
            for i in mmcv.track_iter_progress(range(len(data_infos_ori))):
                out_file = osp.join(out_dir, '%07d.pkl' % i)
                data_infos_path.append(out_file)
                if not osp.exists(out_file):
                    mmcv.dump(data_infos_ori[i], out_file, file_format='pkl')
            data_infos_path = data_infos_path[::self.load_interval]
            return data_infos_path

        return data_infos

    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['bbox2d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def load_annotations_2d(self, ann_file):
        self.coco = COCO(ann_file)
        # self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.METAINFO['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.impath_to_imgid = {}
        self.imgid_to_dataid = {}
        data_infos = []
        total_ann_ids = []
        for i in self.coco.get_img_ids():
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            cam_type = info['camera_type']
            self.impath_to_imgid['data/nuscenes/' 
                                 + 'samples/'
                                 + f"{cam_type}/"
                                 + info['file_name']] = i
            self.imgid_to_dataid[i] = len(data_infos)
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        self.data_infos_2d = data_infos

    def impath_to_ann2d(self, impath):
        img_id = self.impath_to_imgid[impath]
        data_id = self.imgid_to_dataid[img_id]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self.get_ann_info_2d(self.data_infos_2d[data_id], ann_info)
    
    def plot_3d_and_2d_boxes_on_image(
        self,
        image_path,
        boxes_3d,
        bboxes_2d,
        lidar2cam,
        K,
        color_3d=(0, 0, 255),   # Red for 3D projected boxes
        color_2d=(0, 255, 0),   # Green for 2D annotated boxes
        thickness=2,
        save_path="3d_2d_boxes.png"
    ):
        """
        Project 3D bounding boxes and draw them as wireframes,
        overlay 2D GT bboxes in a different color.

        Args:
            image_path: str, path to image
            boxes_3d: LiDARInstance3DBoxes
            bboxes_2d: (N,4) array of 2D boxes [xmin, ymin, xmax, ymax]
            lidar2cam: (4,4) lidar->camera transform
            K: (3,3) camera intrinsics
            color_3d: color for 3D boxes (BGR)
            color_2d: color for 2D boxes (BGR)
            thickness: line thickness
            save_path: output path
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # --- Project 3D boxes ---
        corners = boxes_3d.corners.numpy()  # (N, 8, 3)
        N = corners.shape[0]

        # Homogenize and transform to camera coords
        corners_hom = np.concatenate([corners, np.ones((N, 8, 1))], axis=-1)  # (N,8,4)
        corners_cam = corners_hom @ lidar2cam.T  # (N,8,4)
        corners_cam = corners_cam[..., :3]

        # Keep only boxes in front of camera
        valid_mask = (corners_cam[..., 2] > 0).all(axis=1)
        corners_cam = corners_cam[valid_mask]

        # Project to 2D
        proj = corners_cam @ K.T  # (N,8,3)
        proj_uv = proj[..., :2] / proj[..., 2:3]  # (N,8,2)

        # Define box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
        ]

        # Draw projected 3D boxes
        for box_uv in proj_uv.astype(int):
            for e in edges:
                pt1 = tuple(box_uv[e[0]])
                pt2 = tuple(box_uv[e[1]])
                cv2.line(img, pt1, pt2, color_3d, thickness)

        # --- Draw 2D annotated boxes ---
        for box in bboxes_2d.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color_2d, thickness)

        # Save result
        cv2.imwrite(save_path, img)

        # # Optional matplotlib preview
        # plt.figure(figsize=(12, 8))
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.savefig(save_path.replace(".png", "_plt.png"), bbox_inches="tight", dpi=300)
        # plt.close()
        
    def project_3d_boxes_to_2d(self, gt_bboxes_3d, lidar2cam, K):
        """
        Project 3D boxes into 2D bboxes and get their centers.

        Args:
            gt_bboxes_3d: LiDARInstance3DBoxes
            lidar2cam: (4,4) lidar->camera transform
            K: (3,3) camera intrinsics

        Returns:
            bboxes_2d: (N,4) [xmin, ymin, xmax, ymax]
            centers_2d: (N,2) [cx, cy]
        """
        
        # centers_lidar = self.get_bbox3d_top_center(gt_bboxes_3d).numpy()
        # centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1) #(45, 4)

        # Transform to camera coordinates
        # centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
        # depths = centers_cam[:, 2]

        # Keep only points in front of the camera
        # valid_mask = depths > 0
        # centers_cam = centers_cam[valid_mask]
        # depths = depths[valid_mask]
        # gt_labels_3d_valid = gt_labels_3d[valid_mask]

        # # Project to image plane
        # proj = centers_cam @ K.T
        # proj_uv = proj[:, :2] / proj[:, 2:3]  # (N, 2) pixel coordinates
        
        
        corners = gt_bboxes_3d.corners.numpy()  # (N, 8, 3)

        # Homogenize and transform to camera coords
        corners_hom = np.concatenate([corners, np.ones((corners.shape[0], 8, 1))], axis=-1)  # (N,8,4)
        corners_cam = corners_hom @ lidar2cam.T  # (N,8,4)
        corners_cam = corners_cam[..., :3]

        # Project to image plane
        proj = corners_cam @ K.T  # (N,8,3)
        proj_uv = proj[..., :2] / proj[..., 2:3]  # (N,8,2)

        # Build 2D boxes
        umin = proj_uv[..., 0].min(axis=1)
        umax = proj_uv[..., 0].max(axis=1)
        vmin = proj_uv[..., 1].min(axis=1)
        vmax = proj_uv[..., 1].max(axis=1)
        bboxes_2d = np.stack([umin, vmin, umax, vmax], axis=1)
        centers_2d = np.stack([(umin + umax) / 2, (vmin + vmax) / 2], axis=1)
        
        # depth mask
        depths = corners_cam[:, :, 2].mean(axis=1)
        valid_mask = depths > 0
        # bboxes_2d = bboxes_2d[valid_mask]
        # centers_2d = centers_2d[valid_mask]
        
        return bboxes_2d, centers_2d, valid_mask
    
    
    def project_3d_bbox_to_2d(self, gt_bboxes_3d, lidar2cam, K):
        """
        Project 3D boxes into 2D image bboxes using their corners.

        Args:
            gt_bboxes_3d: LiDARInstance3DBoxes, must have a `.corners` property of shape (N, 8, 3)
            lidar2cam: (4,4) lidar-to-camera transformation
            K: (3,3) camera intrinsics

        Returns:
            bboxes_2d: (M,4) [xmin, ymin, xmax, ymax] in pixels
            centers_2d: (M,2) [cx, cy] in pixels
            valid_mask: (N,) boolean mask indicating boxes in front of camera
        """
        # Get corners
        corners = gt_bboxes_3d.corners.numpy()  # (N, 8, 3)

        # Homogenize corners and transform to camera coordinates
        corners_hom = np.concatenate([corners, np.ones((corners.shape[0], 8, 1))], axis=-1)  # (N,8,4)
        corners_cam = corners_hom @ lidar2cam.T  # (N,8,4)
        corners_cam = corners_cam[..., :3]

        # Compute depth mask (keep boxes with all corners in front of camera)
        depths = corners_cam[:, :, 2]  # (N,8)
        valid_mask = (depths > 0).all(axis=1)
        
        corners_cam = corners_cam[valid_mask]

        # Project corners to image plane
        proj = corners_cam @ K.T  # (M,8,3)
        proj_uv = proj[..., :2] / proj[..., 2:3]  # (M,8,2)

        # Compute 2D bounding boxes from projected corners
        umin = proj_uv[..., 0].min(axis=1)
        umax = proj_uv[..., 0].max(axis=1)
        vmin = proj_uv[..., 1].min(axis=1)
        vmax = proj_uv[..., 1].max(axis=1)
        bboxes_2d = np.stack([umin, vmin, umax, vmax], axis=1)
        
        # Compute 2D centers
        centers_2d = np.stack([(umin + umax)/2, (vmin + vmax)/2], axis=1)

        return bboxes_2d, centers_2d, valid_mask
    
    
    def _map_cam_to_lidar_boxes(self, gt_instances_3d, gt_instances_3d_lidar, extrinsics, tol=0.3):
        """
        Build mapping from (cam_idx, cam_box_idx) → lidar_box_idx
        by comparing transformed box centers.
        
        Args:
            gt_instances_3d: list of length N_cam, each is a list of dicts with 'bbox_3d' (camera coords)
            gt_instances_3d_lidar: list of dicts with 'bbox_3d' (lidar coords)
            extrinsics: list of LiDAR→camera 4x4 matrices, one per camera
            tol: distance threshold in meters for matching centers

        Returns:
            match_dict: {(cam_idx, cam_box_idx): lidar_box_idx}
        """
        match_dict = {}
        lidar_centers = np.array([b['bbox_3d'][:3] for b in gt_instances_3d_lidar])
        
        for cam_idx, cam_boxes in enumerate(gt_instances_3d):
            T_cam_lidar = extrinsics[cam_idx]
            T_lidar_cam = np.linalg.inv(T_cam_lidar)
            
            for cam_box_idx, cam_box in enumerate(cam_boxes):
                cam_center = np.array([*cam_box['bbox_3d'][:3], 1])
                # Convert camera center to lidar frame
                lidar_center_est = (T_lidar_cam @ cam_center)[:3]
                
                # Compute distances to all lidar box centers
                dists = np.linalg.norm(lidar_centers - lidar_center_est, axis=1)
                min_idx = np.argmin(dists)
                
                if dists[min_idx] < tol:
                    match_dict[(cam_idx, cam_box_idx)] = min_idx
                else:
                    match_dict[(cam_idx, cam_box_idx)] = -1  # No match found within tolerance
                    
        return match_dict
    
    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        if not self.load_separate:
            # info = self.data_infos[index]
            info = super().get_data_info(index)
        else:
            # info = mmcv.load(self.data_infos[index], file_format='pkl')
            info = mmcv.load(self.data_list[index], file_format='pkl')
        # standard protocal modified from SECOND.Pytorch
        
        if 'lidar_sweeps' not in info:
            # skip to next sample if no sweep
            return self.get_data_info((index + 1) % len(self))
        
        input_dict = dict(
            sample_idx=info['sample_idx'],
            token=info['token'],
            # pts_filename=info['lidar_path'],
            # sweeps=info['sweeps'],
            pts_filename=info['lidar_points']['lidar_path'],
            sweeps=info['lidar_sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        gt_instances_3d = []
        gt_instances_3d_lidar = info['instances']
        # ego2global = info['ego2global']
        images = {}
        for cam_type, cam_info in info['images'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            img_path = os.path.join(self.data_root, 'samples', cam_type, cam_info['img_path'].split("/")[-1])
            image_paths.append(img_path)
            # obtain lidar to image transformation matrix
            # lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            # lidar2cam_t = cam_info[
            #                   'sensor2lidar_translation'] @ lidar2cam_r.T
            # lidar2cam_rt = np.eye(4)
            # lidar2cam_rt[:3, :3] = lidar2cam_r.T
            # lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_rt = np.array(cam_info['lidar2cam'])
            intrinsic = np.array(cam_info['cam2img'])#['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt)
            intrinsics.append(viewpad)
            extrinsics.append(
                lidar2cam_rt)  ###The extrinsics mean the tranformation from lidar to camera. If anyone want to use the extrinsics as sensor to lidar, please use np.linalg.inv(lidar2cam_rt.T) and modify the ResizeCropFlipImage and LoadMultiViewImageFromMultiSweepsFiles.
            lidar2img_rts.append(lidar2img_rt)
            images[cam_type] = dict(
                img_path=img_path,
                cam2img=cam_info['cam2img'],
                lidar2cam=cam_info['lidar2cam'],
            )
            gt_instance = info['cam_instances'][cam_type]
            # # convert all 3d bboxes to LiDARInstance3DBoxes. currently they are in camera coordinates
            # for i in range(len(gt_instance)):
            #     instance = gt_instance[i]
            #     bbox_3d_cam = instance['bbox_3d']
            #     bbox_3d_lidar = Box3DMode.convert(
            #         CameraInstance3DBoxes(torch.tensor(bbox_3d_cam).unsqueeze(0), box_dim=len(bbox_3d_cam), origin=(0.5, 0.5, 0)),
            #         Box3DMode.CAM,
            #         Box3DMode.LIDAR,
            #         np.linalg.inv(lidar2cam_rt)
            #     )
            #     # make it a LiDARInstance3DBoxes
            #     instance['bbox_3d'] = bbox_3d_lidar.tensor.numpy()
                
            #     gt_instance[i] = instance
            gt_instances_3d.append(gt_instance)
        
        cam_to_lidar_box_mapping = self._map_cam_to_lidar_boxes(gt_instances_3d, gt_instances_3d_lidar, extrinsics, tol=0.3)
        lidar_to_cam_box_mapping = {v: k for k, v in cam_to_lidar_box_mapping.items()}
        
        # for all instances in each camera in gt_instances_3d replace box in camera from to box in lidar frame using cam_to_lidar_box_mapping
        # cam_to_lidar_box_mapping keys = (cam_idx, box_idx), values = lidar_box_idx
        for cam_idx in range(len(gt_instances_3d)):
            for box_idx in range(len(gt_instances_3d[cam_idx])):
                lidar_box_idx = cam_to_lidar_box_mapping[(cam_idx, box_idx)]
                if lidar_box_idx >= 0:  
                    bbox_3d_lidar = gt_instances_3d_lidar[lidar_box_idx]['bbox_3d']
                    gt_instances_3d[cam_idx][box_idx]['bbox_3d'] = bbox_3d_lidar
        
                    
        # now all boxes in gt_instances_3d are in lidar frame
        input_dict.update(
            dict(
                img_timestamp=img_timestamp,
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics, #lidar2cam
                images=images,
                gt_instances_3d=gt_instances_3d,
                gt_instances_3d_lidar=gt_instances_3d_lidar,
                cam_to_lidar_box_mapping=cam_to_lidar_box_mapping,
                lidar_to_cam_box_mapping=lidar_to_cam_box_mapping,
                # ego2global=ego2global,
            ))

        input_dict['img_info'] = info
        # also need to add sample_idx and token for evaluation
        
        if not self.test_mode:
            try:
                annos = self.get_ann_info(index, info)
            except:
                return self.get_data_info((index + 1) % len(self))
            input_dict['ann_info'] = annos

            gt_bboxes_3d = annos['gt_bboxes_3d']  # lidar frame confirmed. tensor([[-7.2906, -8.7071, -1.8431,  4.2570,  1.7260,  1.4890,  0.3434]]))
            gt_labels_3d = annos['gt_labels_3d']
        
        gt_bboxes_2d = []  # per-view 2d bboxes
        gt_bboxes_ignore = []  # per-view 2d bboxes
        gt_bboxes_2d_to_3d = []  # mapping from per-view 2d bboxes to 3d bboxes
        gt_labels_2d = []  # mapping from per-view 2d bboxes to 3d bboxes
        gt_centers2d = []  # per-view 2d centers
        gt_instances_3d = [] 
        
        for cam_i in range(len(image_paths)):
            ann_2d = self.impath_to_ann2d(image_paths[cam_i])
            labels_2d = ann_2d['labels']
            bboxes_2d = ann_2d['bboxes_2d']
            bboxes_ignore = ann_2d['gt_bboxes_ignore']
            # cam_name = list(info['images'].keys())[cam_i]
            # centers_2d = [cam_instance['center_2d'] for cam_instance in info['cam_instances'][cam_name]]
            cam_instances = info['cam_instances'][list(info['images'].keys())[cam_i]] # [-22.61999903633102, -0.6445254496545357, 38.3287087817548, 1.095, 1.78, 0.695, 0.29442787910795093]
            lidar_instances = info['instances']
            # bboxes_cam = ann_2d['bboxes_cam']
            if not self.test_mode:
                gt_instances_3d.append(cam_instances) 
            
            lidar2cam = extrinsics[cam_i]
            K = intrinsics[cam_i][:3, :3]          # camera intrinsics
            
            
            if not self.test_mode:
            # 3D bbox centers in lidar coordinates
                proj_bboxes_2d, proj_centers_2d, valid_mask = self.project_3d_boxes_to_2d(gt_bboxes_3d, lidar2cam, K)
                
                # instead of slicing, keep the same shape
                # optionally mark invalid projections as NaN or some sentinel value
                proj_centers_2d[~valid_mask] = np.nan

                # optionally clip them to in-bounds, but don't slice
                img_info = self.data_infos_2d[self.imgid_to_dataid[self.impath_to_imgid[image_paths[cam_i]]]]
                H, W = img_info['height'], img_info['width']
                in_bounds_mask = (proj_centers_2d[:, 0] >= 0) & (proj_centers_2d[:, 0] < W) & \
                                (proj_centers_2d[:, 1] >= 0) & (proj_centers_2d[:, 1] < H)
                proj_centers_2d[~in_bounds_mask] = np.nan
                # proj_bboxes_2d, proj_centers_2d, valid_mask = self.project_3d_bbox_to_2d(gt_bboxes_3d, lidar2cam, K)
            # self.plot_3d_and_2d_boxes_on_image(
            #     image_paths[cam_i],
            #     gt_bboxes_3d,
            #     bboxes_2d,
            #     lidar2cam,
            #     K,
            #     color_3d=(0, 0, 255),   # red 3D
            #     color_2d=(0, 255, 0),   # green 2D
            #     save_path=f"overlay_cam{cam_i}.png"
            # )

            # centers_lidar = self.get_bbox3d_top_center(gt_bboxes_3d).numpy()
            # centers_lidar_hom = np.concatenate([centers_lidar, np.ones((len(centers_lidar), 1))], axis=1) #(45, 4)

            # Transform to camera coordinates
            # centers_cam = (centers_lidar_hom @ lidar2cam.T)[:, :3]
            # depths = centers_cam[:, 2]

            # Keep only points in front of the camera
            # valid_mask = depths > 0
            # centers_cam = centers_cam[valid_mask]
            # depths = depths[valid_mask]
                # gt_labels_3d_valid = gt_labels_3d[valid_mask]

            # # Project to image plane
            # proj = centers_cam @ K.T
            # proj_uv = proj[:, :2] / proj[:, 2:3]  # (N, 2) pixel coordinates

            # 2D bbox centers
            cx = (bboxes_2d[:, 0] + bboxes_2d[:, 2]) / 2
            cy = (bboxes_2d[:, 1] + bboxes_2d[:, 3]) / 2
            centers_2d = np.stack([cx, cy], axis=-1)
            
            # remove out of bounds proj_centers_2d
            if not self.test_mode:
                img_info = self.data_infos_2d[self.imgid_to_dataid[self.impath_to_imgid[image_paths[cam_i]]]]
                # H, W = img_info['height'], img_info['width']
                # in_bounds_mask = (proj_centers_2d[:, 0] >= 0) & (proj_centers_2d[:, 0] < W) & \
                #                 (proj_centers_2d[:, 1] >= 0) & (proj_centers_2d[:, 1] < H)
                # proj_centers_2d[~in_bounds_mask] = np.nan
            
            # Match projected 3D centers to 2D bbox centers
            # match = self.center_match_2d(centers_2d, proj_uv)
            
                match = self.center_match_2d(centers_2d, proj_centers_2d)
                # image_size = (H, W)
                # match = self.iou_match_2d(bboxes_2d, proj_bboxes_2d, iou_thresh=0.5)
                # match = self.iou_match_2d_clipped(bboxes_2d, proj_bboxes_2d, image_size, iou_thresh=0.5)
                
            # assert (labels_2d[match > -1] == gt_labels_3d_valid[match[match > -1]]).all()
            
            # plot 2d centers and projected 3d centers
            # def plot_centers(image_path, centers_2d, proj_centers_2d, match, save_path="centers.png"):
            #     img = cv2.imread(image_path)
                
            #     for i, c in enumerate(centers_2d.astype(int)):
            #         if match.sum() > 0:
            #             color = (0, 255, 0) if match[i] > -1 else (0, 255, 255)
            #         else:
            #             color = (0, 255, 255)
            #         cv2.circle(img, tuple(c), 5, color, -1)
            #     for i, c in enumerate(proj_centers_2d.astype(int)):
            #         cv2.circle(img, tuple(c), 3, (255, 0, 0), -1)
            #     cv2.imwrite(save_path, img)
                # plt.figure(figsize=(12, 8))
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.axis("off")
                # plt.savefig(save_path.replace(".png", "_plt.png"), bbox_inches="tight", dpi=300)
                # plt.close()
            # plot_centers(image_paths[cam_i], centers_2d, proj_centers_2d, match, save_path=f"centers_cam{cam_i}_idx{index}.png")

            gt_bboxes_2d.append(bboxes_2d)
            if not self.test_mode:
                gt_bboxes_2d_to_3d.append(match)
            gt_labels_2d.append(labels_2d)
            gt_bboxes_ignore.append(bboxes_ignore)
            gt_centers2d.append(centers_2d)
        
        if self.test_mode:
            annos = dict()
        annos['gt_bboxes'] = gt_bboxes_2d
        annos['gt_bboxes_labels'] = gt_labels_2d
        if not self.test_mode:
            annos['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d
        annos['gt_bboxes_ignore'] = gt_bboxes_ignore
        annos['centers_2d'] = gt_centers2d
        if not self.test_mode:
            annos['gt_instances_3d'] = gt_instances_3d
        annos['gt_bboxes_3d'] = gt_bboxes_3d
        # gt_instances_3d in cam frame. use mapping to get lidar frame boxes
        for cam_idx in range(len(gt_instances_3d)):
            for box_idx in range(len(gt_instances_3d[cam_idx])):
                lidar_box_idx = cam_to_lidar_box_mapping[(cam_idx, box_idx)]
                if lidar_box_idx >= 0 and lidar_box_idx < len(gt_bboxes_3d):
                    bbox_3d_lidar = gt_bboxes_3d.tensor[lidar_box_idx].numpy()
                    gt_instances_3d[cam_idx][box_idx]['bbox_3d'] = bbox_3d_lidar 
        annos['gt_instances_3d_lidar'] = gt_instances_3d_lidar
        
        # replace gt_instances_3d in annos with those in lidar frame
        annos['gt_instances_3d'] = gt_instances_3d
        
        # data_list = []
        # data_info = super().parse_data_info(info)
        # for idx, (cam_id, img_info) in enumerate(data_info['images'].items()):
        #     num_cameras = 6
        #     data_info['sample_idx'] = data_info['sample_idx'] * num_cameras + idx
        #     data_info['token'] = data_info['token']
        #     data_info['ego2global'] = data_info['ego2global']

        #     if not self.test_mode:
        #         # used in traing
        #         data_info['ann_info'] = self.parse_ann_info(data_info)
        #     if self.test_mode and self.load_eval_anns:
        #         data_info['eval_ann_info'] = \
        #             self.parse_ann_info(data_info)
        #     data_list.append(data_info)


        # add sample_idx and token for evaluation
        input_dict['sample_idx'] = info['sample_idx']
        input_dict['token'] = info['token']
        
        # input_dict['images']['CAM_FRONT']['lidar2cam']
        lidar2cam = [np.array(cam_info['lidar2cam']) for cam_type, cam_info in info['images'].items()]
        
        input_dict['lidar2cam'] = lidar2cam
        
        # 2d bboxes gt. make a list of InstanceData for each view
        # add blank bboxes labels centers_2d as default
        gt_instances = [InstanceData(
            bboxes=torch.tensor([]).float(),
            labels=torch.tensor([]).long(),
            # bboxes_2d_to_3d=torch.tensor([]).long(),
            centers_2d=torch.tensor([]).float(),
            ) for _ in range(len(image_paths))]
        for cam_i in range(len(image_paths)):
            gt_instances[cam_i].bboxes = annos['gt_bboxes'][cam_i]
            gt_instances[cam_i].labels = annos['gt_bboxes_labels'][cam_i]
            if not self.test_mode:
                gt_instances[cam_i].bboxes_2d_to_3d = annos['gt_bboxes_2d_to_3d'][cam_i]
            gt_instances[cam_i].centers_2d = annos['centers_2d'][cam_i]
            
        input_dict['gt_instances'] = gt_instances
        input_dict['ann_info'] = annos
        
        
        # gt_instances_3d_lidar
        
        return input_dict

    # def center_match_2d(self, pts_a, pts_b, thresh=50):
    #     """
    #     Match 2D centers by nearest neighbor in pixel space.
    #     pts_a: (N, 2) array (e.g., annotated 2D centers)
    #     pts_b: (M, 2) array (e.g., projected 3D centers)
    #     Returns: (N,) array of indices into pts_b, or -1 if no match
    #     """
    #     if len(pts_a) == 0 or len(pts_b) == 0:
    #         return np.full(len(pts_a), -1, dtype=np.int32)

    #     dist = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)  # (N, M)
    #     match = dist.argmin(1)
    #     match[dist.min(1) > thresh] = -1  # reject if too far
    #     assert len(match) == len(pts_a)
    #     return match
    def center_match_2d(self, pts_a, pts_b, thresh=50):
        if len(pts_a) == 0:
            matches = np.array([], dtype=np.int32)
            assert len(matches) == len(pts_a)
            return matches
        if len(pts_b) == 0:
            matches = np.full(len(pts_a), -1, dtype=np.int32)
            assert len(matches) == len(pts_a)
            return matches

        valid_mask_b = ~np.isnan(pts_b).any(axis=1)
        pts_b_valid = pts_b[valid_mask_b]
        if len(pts_b_valid) == 0:
            matches = np.full(len(pts_a), -1, dtype=np.int32)
            assert len(matches) == len(pts_a)
            return matches

        dist = np.linalg.norm(pts_a[:, None, :] - pts_b_valid[None, :, :], axis=-1)
        match_local = dist.argmin(1)
        match_local[dist.min(1) > thresh] = -1

        # Map back to global 3D indices
        global_indices = np.arange(len(pts_b))[valid_mask_b]
        matches = np.full(len(pts_a), -1, dtype=np.int32)
        matches[match_local != -1] = global_indices[match_local[match_local != -1]]
        assert len(matches) == len(pts_a)
        return matches

    def iou_match_2d(self, boxes_a, boxes_b, iou_thresh=0.5):
        """
        Match boxes_a to boxes_b by IoU.

        Args:
            boxes_a: (N,4) ground-truth 2D boxes [xmin, ymin, xmax, ymax] (numpy or tensor)
            boxes_b: (M,4) projected 3D boxes [xmin, ymin, xmax, ymax] (numpy or tensor)
            iou_thresh: minimum IoU to accept a match

        Returns:
            matches: (N,) indices into boxes_b, -1 if no match
        """
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.full(len(boxes_a), -1, dtype=np.int32)

        # Convert to torch tensors
        boxes_a = torch.as_tensor(boxes_a, dtype=torch.float32)
        boxes_b = torch.as_tensor(boxes_b, dtype=torch.float32)

        # Compute IoU matrix
        iou_matrix = box_iou(boxes_a, boxes_b)  # (N, M)

        # Pick best match for each GT box
        max_iou, match_idx = iou_matrix.max(dim=1)
        match_idx[max_iou < iou_thresh] = -1  # reject low IoU matches

        return match_idx.cpu().numpy().astype(np.int32)
    
    def iou_match_2d_clipped(self, boxes_a, boxes_b, image_size, iou_thresh=0.5):
        """
        Match 2D boxes by IoU, clipping boxes_b to image boundaries first.

        Args:
            boxes_a: (N,4) GT boxes [xmin, ymin, xmax, ymax] (numpy or tensor)
            boxes_b: (M,4) projected 3D boxes [xmin, ymin, xmax, ymax] (numpy or tensor)
            image_size: tuple (H, W) of image dimensions
            iou_thresh: minimum IoU to accept a match

        Returns:
            matches: (N,) indices into boxes_b, -1 if no match
        """
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.full(len(boxes_a), -1, dtype=np.int32)

        H, W = image_size

        # Convert to torch tensors
        boxes_a = torch.as_tensor(boxes_a, dtype=torch.float32)
        boxes_b = torch.as_tensor(boxes_b, dtype=torch.float32)

        # Clip projected boxes to image boundaries
        boxes_b_clipped = boxes_b.clone()
        boxes_b_clipped[:, 0] = boxes_b_clipped[:, 0].clamp(0, W-1)
        boxes_b_clipped[:, 1] = boxes_b_clipped[:, 1].clamp(0, H-1)
        boxes_b_clipped[:, 2] = boxes_b_clipped[:, 2].clamp(0, W-1)
        boxes_b_clipped[:, 3] = boxes_b_clipped[:, 3].clamp(0, H-1)

        # Remove boxes that are fully outside the image
        widths = boxes_b_clipped[:, 2] - boxes_b_clipped[:, 0]
        heights = boxes_b_clipped[:, 3] - boxes_b_clipped[:, 1]
        valid_mask = (widths > 0) & (heights > 0)
        boxes_b_clipped = boxes_b_clipped[valid_mask]

        if boxes_b_clipped.shape[0] == 0:
            return np.full(len(boxes_a), -1, dtype=np.int32)

        # Compute IoU
        iou_matrix = box_iou(boxes_a, boxes_b_clipped)

        # Best match per GT box
        max_iou, match_idx = iou_matrix.max(dim=1)
        match_idx[max_iou < iou_thresh] = -1

        # Map back to original indices
        valid_indices = torch.nonzero(valid_mask).squeeze(1)
        match_idx = match_idx.numpy()
        final_match = np.full(len(boxes_a), -1, dtype=np.int32)
        for i, m in enumerate(match_idx):
            if m != -1:
                final_match[i] = valid_indices[m].item()

        return final_match

    def center_match(self, bboxes_a, bboxes_b):
        cts_a, cts_b = bboxes_a[:, :3], bboxes_b[:, :3]
        if len(cts_a) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        if len(cts_b) == 0:
            return np.zeros(len(cts_a), dtype=np.int32) - 1
        dist = np.abs(cts_a[:, None] - cts_b[None]).sum(-1)
        match = dist.argmin(1)
        match[dist.min(1) > 1e-3] = -1
        return match

    def get_ann_info(self, index, info=None):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # if not self.load_separate:
        #     info = self.data_infos[index]
        # else:
        #     info = mmcv.load(self.data_infos[index], file_format='pkl')
        if info is None:
            if not self.load_separate:
                info = super().get_data_info(index)
                info = info if isinstance(info, dict) else info['img_info']
            else:
                info = mmcv.load(self.data_infos[index], file_format='pkl')
        
        ann = info['ann_info']
        
        # filter out bbox containing no points
        if self.use_valid_flag and 'bbox_3d_isvalid' in ann:
            # mask = info['valid_flag']
            mask = ann['bbox_3d_isvalid']
        elif 'num_lidar_pts' in ann:
            mask = ann['num_lidar_pts'] > 0
        # else:
        #     mask = info['num_lidar_pts'] > 0
        else:
            mask = np.ones(len(ann['gt_bboxes_3d']), dtype=bool)

        # gt_bboxes_3d = info['gt_boxes'][mask]
        # gt_names_3d = info['gt_names'][mask]
        gt_bboxes_3d = ann['gt_bboxes_3d'][mask].numpy()
        gt_labels_3d = ann['gt_labels_3d'][mask]
        gt_names_3d  = ann['gt_bboxes_labels'][mask]  # string names

        # gt_labels_3d = []
        # for cat in gt_names_3d:
        #     if cat in self.CLASSES:
        #         gt_labels_3d.append(self.CLASSES.index(cat))
        #     else:
        #         gt_labels_3d.append(-1)
        # gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity and 'velocities' in ann:
            gt_velocity = ann['velocities'][mask]#info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def get_ann_info_2d(self, img_info_2d, ann_info_2d):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,
                depths, bboxes_ignore, masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        for i, ann in enumerate(ann_info_2d):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            
            # scale from normalized to absolute pixel units
            # only do if bboxes are normalized
            if x1 <= 1.0 and y1 <= 1.0 and w <= 1.0 and h <= 1.0 and x1 >= 0.0 and y1 >= 0.0 and w >= 0.0 and h >= 0.0:
                is_normed = True
            else:
                is_normed = False
            if is_normed:
                x1 *= img_info_2d['width']
                y1 *= img_info_2d['height']
                w  *= img_info_2d['width']
                h  *= img_info_2d['height']
            
            inter_w = max(0, min(x1 + w, img_info_2d['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info_2d['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                # bbox_cam3d = np.concatenate([bbox_cam3d], axis=-1)
                # gt_bboxes_cam3d.append(bbox_cam3d.squeeze())

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, 6), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            # bboxes_cam=gt_bboxes_cam3d,
            bboxes_2d=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            labels=gt_labels, )
        return ann

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name in ['pts_bbox', 'img_bbox']:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(jsonfile_prefix, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox',
                         ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)

        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,):

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None and not isinstance(tmp_dir, str):
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict
    
    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == 'mv_image_based':
            data_list = []
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])

            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_list = []
            data_info = super().parse_data_info(info)
            for idx, (cam_id, img_info) in enumerate(data_info['images'].items()):
                num_cameras = 6
                data_info['sample_idx'] = data_info['sample_idx'] * num_cameras + idx
                data_info['token'] = data_info['token']
                data_info['ego2global'] = data_info['ego2global']

                if not self.test_mode:
                    # used in traing
                    data_info['ann_info'] = self.parse_ann_info(data_info)
                if self.test_mode and self.load_eval_anns:
                    data_info['eval_ann_info'] = \
                        self.parse_ann_info(data_info)
                data_list.append(data_info)
            
            return data_list


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def visualize_2d_with_3d_distance(image_paths, gt_bboxes_2d, gt_bboxes_3d, gt_bboxes_2d_to_3d, out_dir="debug_2d_3d_vis"):
    os.makedirs(out_dir, exist_ok=True)

    def compute_distance_from_box_obj(boxes_3d, idx):
       # Get the center as a tensor, then move to CPU/NumPy
       box_tensor = boxes_3d.tensor[idx].cpu().numpy()
       x, y, z = box_tensor[:3]
       return math.sqrt(x**2 + y**2 + z**2)


    for img_idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_path} (couldn't load image).")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes_2d = gt_bboxes_2d[img_idx]
        map_2d_to_3d = gt_bboxes_2d_to_3d[img_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(img)
        ax.set_title(f"Image {img_idx}")

        for i, bbox_2d in enumerate(bboxes_2d):
            x1, y1, x2, y2 = bbox_2d
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            if i < len(map_2d_to_3d):
                idx_3d = map_2d_to_3d[i]
                if idx_3d != -1:
                    dist = compute_distance_from_box_obj(gt_bboxes_3d, idx_3d)
                    ax.text(x1, y1 - 5, f"{dist:.2f}m", color='yellow', fontsize=10, weight='bold')

        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"vis_{img_idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization: {save_path}")
# # assuming everything’s already defined
# visualize_2d_with_3d_distance(
#     image_paths=image_paths,
#     gt_bboxes_2d=gt_bboxes_2d,
#     gt_bboxes_3d=gt_bboxes_3d,
#     gt_bboxes_2d_to_3d=gt_bboxes_2d_to_3d,
#     out_dir="debug_vis_dataloader"
# )