import mmcv
import numpy as np
# from mmdet.datasets.pipelines import to_tensor
from mmcv.transforms import to_tensor
# from mmdet3d.datasets import PIPELINES
from mmdet3d.registry import TRANSFORMS
# from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D, Collect3D
# from mmdet.core.visualization.image import imshow_det_bboxes, imshow_gt_det_bboxes
from projects.MV2D.mmdet3d_plugin.core.bbox.visualization.image import imshow_det_bboxes, imshow_gt_det_bboxes
# from mmdet3d.core.visualizer import show_multi_modality_result
# from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
import trimesh
from os import path as osp


# import numpy as np
# from mmcv.parallel import DataContainer as DC

from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
from mmdet3d.structures.points import BasePoints
# from mmdet.datasets.builder import PIPELINES
# from mmdet.datasets.pipelines import to_tensor

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs

@TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img for img in results['img']]  # .transpose(2, 0, 1)
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = to_tensor(imgs)#DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img']) # .transpose(2, 0, 1)
                results['img'] = to_tensor(img)#DC(to_tensor(img), stack=True)
        for key in [
                'proposals', 'gt_bboxes_2d', 'gt_bboxes_ignore', 'gt_labels_2d',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]#DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = to_tensor(results[key])#DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = results['gt_bboxes_3d']#DC(results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
                #DC(to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = results['gt_masks'] #DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(results['gt_semantic_seg'][None, ...])
            # DC(to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class Collect3D(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'token', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'token'
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'batch_input_shape', 'ego2global', 'lidar2cam'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = img_metas#DC(img_metas, cpu_only=True)
        
        for key in self.keys:
            # if key not in results:
            #     data[key] = results['ann_info'][key]
            # else:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@TRANSFORMS.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = results['points'].tensor#DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])#DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str




def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode,
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    # elif box_mode == 'lidar':
    #     draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))

@TRANSFORMS.register_module()
class DefaultFormatBundleMono3D(DefaultFormatBundle3D):

    def __call__(self, results):
        results = super(DefaultFormatBundleMono3D, self).__call__(results)
        for key in [
                'gt_bboxes_2d', 'gt_labels_2d', 'gt_bboxes_2d_to_3d', 'centers_2d',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]#DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = to_tensor(results[key])#DC(to_tensor(results[key]))
        return results


@TRANSFORMS.register_module()
class CollectMono3D(Collect3D):
    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'token',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'intrinsics', 'extrinsics', 'timestamp', 'ego2global',
                   ),
        debug=False,
        classes=('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle',
                 'pedestrian', 'traffic_cone', 'barrier', 'ignore'),
    ):
        super(CollectMono3D, self).__init__(keys, meta_keys)
        self.debug = debug
        self.classes = classes

    @staticmethod
    def parse_img_metas(img_metas):
        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        num_views = len(img_metas['img_shape'])
        img_metas_views = img_metas
        img_metas = []
        for j in range(num_views):
            img_meta = dict()
            for k, v in img_metas_views.items():
                if isinstance(v, list):
                    img_meta[k] = v[j]
                elif k == 'ori_shape':
                    img_meta[k] = v[:3]
                else:
                    img_meta[k] = v
            img_metas.append(img_meta)
        return img_metas

    @staticmethod
    def denormalize(img, img_norm_config):
        img = img.permute(1, 2, 0).numpy()
        img = mmcv.imdenormalize(img, img_norm_config['mean'], img_norm_config['std'], img_norm_config['to_rgb'])
        return img

    @staticmethod
    def get_box_params(bboxes, intrinsics, extrinsics, roi_size):
        import torch
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = torch.from_numpy(intrinsic).to(bbox.device).type(bbox.dtype)
            extrinsic = torch.from_numpy(extrinsic).to(bbox.device).type(bbox.dtype)
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            wh_bbox = bbox[:, 2:4] - bbox[:, :2]
            wh_roi = wh_bbox.new_tensor(roi_size)
            scale = wh_roi[None] / wh_bbox
            intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
            intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list

    def __call__(self, results):
        results = super(CollectMono3D, self).__call__(results)
        if self.debug:
            vis_2d = True
            vis_3d = True
            vis_bbox = True
            img_metas = self.parse_img_metas(results['img_metas'])
            for img_id, (img, img_meta) in enumerate(zip(results['img'].data, img_metas)):
                img = self.denormalize(img, img_meta['img_norm_cfg'])
                img_3d, img_bbox = img.copy(), img.copy()
                file_name = 'debug/' + '/'.join(img_meta['filename'].split('/')[-2:])
                file_name_3d = file_name.replace('.jpg', '_gt3d.jpg')
                prefix_bbox = 'debug/bbox/' + '/'.join(img_meta['filename'].split('/')[-2:]).replace('.jpg', '')

                # problem img: CAM_BACK/n008-2018-09-18-14-35-12-0400__CAM_BACK__1537295917937558.jpg
                if vis_2d:
                    bboxes_2d = results['gt_bboxes_2d'].data[img_id].numpy()
                    labels_2d = results['gt_labels_2d'].data[img_id].numpy()
                    img = imshow_det_bboxes(
                        img,
                        bboxes_2d,
                        labels_2d,
                        class_names=self.classes,
                        bbox_color='green',
                        text_color='green',
                        show=False,
                        out_file=None)
                    bboxes_ignore = results['gt_bboxes_ignore'].data[img_id].numpy()
                    labels_ignore = np.zeros(len(bboxes_ignore), dtype=np.int32) + len(self.classes) - 1
                    img = imshow_det_bboxes(
                        img,
                        bboxes_ignore,
                        labels_ignore,
                        class_names=self.classes,
                        bbox_color='red',
                        text_color='red',
                        show=False,
                        out_file=None)
                    mmcv.imwrite(img, file_name)

                if vis_3d:
                    gt_ids = results['gt_bboxes_2d_to_3d'].data[img_id].unique()
                    gt_ids = gt_ids[gt_ids > -1].long()
                    bboxes_3d = results['gt_bboxes_3d'].data[gt_ids]
                    # lidar2img = img_meta['intrinsics'] @ img_meta['extrinsics'].T
                    lidar2img = img_meta['lidar2img']
                    # img_3d = draw_lidar_bbox3d_on_img(bboxes_3d, img_3d, lidar2img, None)
                    mmcv.imwrite(img_3d, file_name_3d)

                if vis_bbox:
                    bboxes_2d = results['gt_bboxes_2d'].data[img_id]
                    gt_ids = results['gt_bboxes_2d_to_3d'].data[img_id].long()
                    bboxes_2d = bboxes_2d[gt_ids > -1]
                    gt_ids = gt_ids[gt_ids > -1]
                    bboxes_3d = results['gt_bboxes_3d'].data[gt_ids]

                    roi_size = (40, 40)
                    intrinsics, extrinsics = self.get_box_params(
                        [bboxes_2d.int().float()], [img_meta['intrinsics']], [img_meta['extrinsics']], roi_size)

                    for i in range(len(bboxes_2d)):
                        b2d = bboxes_2d[i]
                        b3d = bboxes_3d[i:i+1]
                        intrins = intrinsics[i]
                        extrins = extrinsics[i]

                        crop = b2d.int().numpy()
                        img_crop = img_bbox[crop[1]:crop[3], crop[0]:crop[2]].copy()
                        img_crop = mmcv.imresize(img_crop, roi_size)
                        lidar2img = (intrins @ extrins.T).numpy()
                        # img_crop = draw_lidar_bbox3d_on_img(b3d, img_crop, lidar2img, None)
                        mmcv.imwrite(img_crop, prefix_bbox + '%03d.jpg' % i)

            import ipdb; ipdb.set_trace()

        return results
    