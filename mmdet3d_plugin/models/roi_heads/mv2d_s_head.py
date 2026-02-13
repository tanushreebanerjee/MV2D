# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from mmdet.structures.bbox import bbox2roi
# from mmdet.models.builder import HEADS
from .mv2d_head import MV2DHead

from mmdet3d.registry import MODELS as MODELS_3D
from projects.MV2D.mmdet3d_plugin.models.builder import HEADS


# @MODELS_3D.register_module()
@HEADS.register_module()
class MV2DSHead(MV2DHead):
    def __init__(self,
                 # denoise setting
                 use_denoise=False,
                 neg_bbox_loss=False,
                 denoise_scalar=10,
                 denoise_noise_scale=1.0,
                 denoise_noise_trans=0.0,
                 denoise_weight=1.0,
                 denoise_split=0.75,
                 **kwargs):
        super(MV2DSHead, self).__init__(**kwargs)
        self.use_denoise = use_denoise
        self.neg_bbox_loss = neg_bbox_loss
        self.denoise_scalar = denoise_scalar
        self.denoise_noise_scale = denoise_noise_scale
        self.denoise_noise_trans = denoise_noise_trans
        self.denoise_weight = denoise_weight
        self.denoise_split = denoise_split

    def prepare_for_dn(self, batch_size, reference_points, img_metas, ref_num, eps=1e-4):
        if self.training:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d'].gravity_center, img_meta['gt_bboxes_3d'].tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d'] for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.denoise_scalar, 1).view(-1)
            known_labels = labels.repeat(self.denoise_scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.denoise_scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.denoise_scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.denoise_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.denoise_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.denoise_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0 + eps, max=1.0 - eps)
                mask = torch.norm(rand_prob, 2, 1) > self.denoise_split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.denoise_scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size,
                                                                                                             1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat(
                    [map_known_indice + single_pad * i for i in range(self.denoise_scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + ref_num
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.denoise_scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.denoise_scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _bbox_forward_denoise(self, x, proposal_list, img_metas):
        # avoid empty 2D detection
        if sum([len(p) for p in proposal_list]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 1.0, 0]], dtype=proposal_list[0].dtype,
                                    device=proposal_list[0].device)
            proposal_list = [proposal] + proposal_list[1:]
        

        for i in range(len(proposal_list)):
            proposal_i = proposal_list[i]
            if proposal_i.shape[1] == 5 and proposal_i.shape[0]==0:
                proposal_i = torch.zeros((0, 4), dtype=proposal_i.dtype, device=proposal_i.device)
                proposal_list[i] = proposal_i
                

        rois = bbox2roi(proposal_list)
        # DEBUG: rois are built from proposal_list; shapes/types may vary
        # intrinsics, extrinsics = self.get_box_params(proposal_list,
        #                                              [img_meta['intrinsics'] for img_meta in img_metas],
        #                                              [img_meta['extrinsics'] for img_meta in img_metas])
        intrinsics, extrinsics, lidar2img = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas],
                                                     [img_meta['lidar2img'] for img_meta in img_metas]
                                                     )
        # Debug: optionally emit small diagnostics when `MV2D_DEBUG=1`.
        # if os.environ.get('MV2D_DEBUG') == '1':
        #     try:
        #         print('MV2D_DEBUG _bbox_forward_denoise: rois.shape=', rois.shape)
        #         print('MV2D_DEBUG _bbox_forward_denoise: intrinsics.shape=', intrinsics.shape)
        #         print('MV2D_DEBUG _bbox_forward_denoise: lidar2img.shape=', lidar2img.shape)
        #     except Exception:
        #         pass
        
        # if rois.shape[1] == 6: (rois = view_id, class, x1, y1, x2, y2)
        # we need it to be [view_idx, x1, y1, x2, y2]
        if rois.shape[1] == 6:
            view_ids = rois[:, 0:1]
            class_ids = rois[:, 1:2]
            bboxs = rois[:, 2:]
            rois = torch.cat([view_ids, bboxs], dim=1)
        
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        # 3dpe was concatenated to fpn feature
        c = bbox_feats.size(1)
        bbox_feats, pe = bbox_feats.split([c // 2, c // 2], dim=1)

        # intrinsics as extra input feature
        extra_feats = dict(
            intrinsic=self.process_intrins_feat(rois, intrinsics)
        ) 

        # query generator
        reference_points, return_feats = self.query_generator(bbox_feats, intrinsics, extrinsics, lidar2img, extra_feats)
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.pc_range[0]) / (
                self.pc_range[3] - self.pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.pc_range[1]) / (
                self.pc_range[4] - self.pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.pc_range[2]) / (
                self.pc_range[5] - self.pc_range[2])
        reference_points.clamp(min=0, max=1)

        # generate box correlation
        corr, mask = self.box_corr_module.gen_box_roi_correlation(rois, [len(p) for p in proposal_list], img_metas)

        if self.use_denoise and self.training:
            # bbox_feats: [num_rois, c, h, w]
            n_rois, c, h, w = bbox_feats.shape
            cross_attn_mask = bbox_feats.new_ones((n_rois, n_rois + 1)).bool()
            corr[~mask] = n_rois  # [num_rois, max_corr]
            cross_attn_mask = torch.scatter(cross_attn_mask, 1, corr, 0)
            cross_attn_mask = cross_attn_mask[:, :n_rois, None, None].expand(n_rois, n_rois, h, w)

            reference_points_ori = reference_points
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(1, reference_points, img_metas[0:1],
                                                                         len(reference_points))
            reference_points = reference_points[0]
            cross_attn_mask_pad = cross_attn_mask.new_zeros(
                (len(reference_points) - len(reference_points_ori), n_rois, h, w))
            cross_attn_mask = torch.cat([cross_attn_mask_pad, cross_attn_mask])

            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[None],
                                                            bbox_feats[None],
                                                            torch.zeros_like(bbox_feats[None, :, 0]).bool(),
                                                            pe[None],
                                                            attn_mask=attn_mask,
                                                            cross_attn_mask=cross_attn_mask,
                                                            force_fp32=self.force_fp32, )
        else:
            mask_dict = None

            corr_feats = bbox_feats[corr]  # [num_rois, num_corrs, c, h, w]
            corr_pe = pe[corr]
            all_cls_scores, all_bbox_preds = self.bbox_head(reference_points[:, None],
                                                            corr_feats,
                                                            ~mask[..., None, None].expand_as(corr_feats[:, :, 0]),
                                                            corr_pe,
                                                            attn_mask=None,
                                                            cross_attn_mask=None,
                                                            force_fp32=self.force_fp32, )

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]

        cls_scores, bbox_preds = [], []
        for c, b in zip(all_cls_scores, all_bbox_preds):
            cls_scores.append(c.flatten(0, 1))
            bbox_preds.append(b.flatten(0, 1))

        bbox_results = dict(
            cls_scores=cls_scores, bbox_preds=bbox_preds, bbox_feats=bbox_feats, return_feats=return_feats,
            intrinsics=intrinsics, extrinsics=extrinsics, rois=rois, dn_mask_dict=mask_dict,
        )

        return bbox_results

    def _bbox_forward(self, x, proposal_list, img_metas):
        bbox_results = self._bbox_forward_denoise(x, proposal_list, img_metas)
        return bbox_results

    def prepare_for_dn_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      ori_gt_bboxes_3d,
                      ori_gt_labels_3d,
                      attr_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        assert len(img_metas) // img_metas[0]['num_views'] == 1

        num_imgs = len(img_metas)

        proposal_boxes = []
        proposal_scores = []
        proposal_classes = []
        for i in range(num_imgs):
            proposal_boxes.append(proposal_list[i][:, :4]) # 6
            proposal_scores.append(proposal_list[i][:, 4])
            proposal_classes.append(proposal_list[i][:, 5])

        # position encoding
        pos_enc = self.position_encoding(x, img_metas)
        x = [torch.cat([feat, pe], dim=1) for feat, pe in zip(x, pos_enc)]

        losses = dict()

        if self.use_denoise:
            img_metas[0]['gt_bboxes_3d'] = ori_gt_bboxes_3d[0]
            img_metas[0]['gt_labels_3d'] = ori_gt_labels_3d[0]

        results_from_last = self._bbox_forward_train(x, proposal_boxes, img_metas)
        preds = results_from_last['pred']

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        loss_weights = copy.deepcopy(self.stage_loss_weights)

        # use the matching results from last stage for loss calculation
        loss_stage = []
        num_layers = len(cls_scores)
        for layer in range(num_layers):
            loss_bbox = self.bbox_head.loss(
                ori_gt_bboxes_3d, ori_gt_labels_3d, {'cls_scores': [cls_scores[num_layers - 1 - layer]],
                                                     'bbox_preds': [bbox_preds[num_layers - 1 - layer]]},
            )
            loss_stage.insert(0, loss_bbox)

        if results_from_last.get('dn_mask_dict', None) is not None:
            dn_mask_dict = results_from_last['dn_mask_dict']
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_dn_loss(
                dn_mask_dict)
            for i in range(len(output_known_class)):
                dn_loss_cls, dn_loss_bbox = self.bbox_head.dn_loss_single(
                    output_known_class[i], output_known_coord[i], known_bboxs, known_labels, num_tgt,
                    self.pc_range, self.denoise_split, neg_bbox_loss=self.neg_bbox_loss
                )
                losses[f'l{i}.dn_loss_cls'] = dn_loss_cls * self.denoise_weight * loss_weights[i]
                losses[f'l{i}.dn_loss_bbox'] = dn_loss_bbox * self.denoise_weight * loss_weights[i]

        for layer in range(num_layers):
            lw = loss_weights[layer]
            for k, v in loss_stage[layer].items():
                losses[f'l{layer}.{k}'] = v * lw if 'loss' in k else v

        return losses

    def loss(self, x, proposal_list, gt_bboxes_3d, gt_labels_3d, img_metas, **kwargs):
        """
        Args:
            x (list[Tensor]): Multi-level features from the backbone.
            proposal_list (list[Tensor]): List of region proposals.
            gt_bboxes_3d (list[BaseInstance3DBoxes]): Ground truth 3D boxes.
            gt_labels_3d (list[Tensor]): Ground truth labels for 3D boxes.
            img_metas (list[dict]): Meta information of images.
            **kwargs: Other arguments.

        Returns:
            dict: A dictionary of loss components.
        """
        return self.forward_train(x, proposal_list, gt_bboxes_3d, gt_labels_3d, img_metas, **kwargs)

def debug_rescaled_intrinsics(
    K_full,
    K_roi,
    bbox,
    roi_size,
    device
):
    torch.set_printoptions(precision=4, sci_mode=False)

    dtype = K_full.dtype

    X_cam = torch.tensor([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [1.0, 1.0, 10.0],
    ], device=device, dtype=dtype)

    bbox = bbox.to(device=device, dtype=dtype)
    roi_size = torch.tensor(roi_size, device=device, dtype=dtype)

    uv_full = project_points(K_full, X_cam)

    scale = roi_size / (bbox[2:4] - bbox[:2])
    uv_expected = (uv_full - bbox[:2]) * scale

    uv_roi = project_points(K_roi, X_cam)

    
    print("\n==== ROI INTRINSIC DEBUG ====")
    print("Full image projection:\n", uv_full)
    print("Expected ROI projection:\n", uv_expected)
    print("Actual ROI intrinsic projection:\n", uv_roi)
    print("Diff (actual - expected):\n", uv_roi - uv_expected)


def project_points(K, X_cam):
    """
    K: [3,3] or [4,4]
    X_cam: [N,3]
    """
    if K.shape[0] == 4:
        K = K[:3, :3]

    dtype = K.dtype
    device = K.device

    X_cam = X_cam.to(device=device, dtype=dtype)

    X = X_cam.t()          # [3, N]
    x = K @ X              # [3, N]
    x = x[:2] / x[2:3]
    return x.t()

def unproject_points(K, uv, depth):
    """
    K: [3,3] or [4,4]
    uv: [N,2]
    depth: [N]
    """
    if K.shape[0] == 4:
        K = K[:3, :3]

    dtype = K.dtype
    device = K.device

    uv = uv.to(device=device, dtype=dtype)
    depth = depth.to(device=device, dtype=dtype)

    uv_h = torch.cat([uv, torch.ones((uv.shape[0], 1), device=device, dtype=dtype)], dim=1)  # [N,3]
    K_inv = torch.inverse(K)

    X_cam = (K_inv @ uv_h.t()).t()  # [N,3]
    X_cam = X_cam * depth.unsqueeze(1)
    return X_cam
    
def debug_intrinsics(
    proposal,
    intrinsics_full,
    intrinsics_roi,
    roi_size,
    device
):
    # take center of patch = 0, 0 in ROI, center of proposal bbox in nominal image coords
    # project to cam coords, use K_roi^-1 for roi center, K_full^-1 for full image center
    # should be same
    # roi_center = 0, 0 in roi coords
    
    torch.set_printoptions(precision=4, sci_mode=False)

    dtype = intrinsics_full.dtype
    proposal = proposal.to(device=device, dtype=dtype)
    intrinsics_full = intrinsics_full.to(device=device, dtype=dtype)
    intrinsics_roi = intrinsics_roi.to(device=device, dtype=dtype)
    roi_size = torch.tensor(roi_size, device=device, dtype=dtype)
    
    uv_roi = torch.tensor([[roi_size[0]/2, roi_size[1]/2]], device=device, dtype=dtype)  # [1,2]
    depth = torch.tensor([10.0], device=device, dtype=dtype)  #
    X_cam_from_roi = unproject_points(intrinsics_roi, uv_roi, depth)  # [1,3]
    print("X_cam from ROI intrinsic:", X_cam_from_roi)
    # now project to full image coords
    uv_full = project_points(intrinsics_full, X_cam_from_roi)  # [1,2]
    print("Projected uv in full image:", uv_full)
    # should be center of proposal
    proposal_center = (proposal[0:2] + proposal[2:4]) / 2
    print("Proposal center:", proposal_center)
    print("Diff (proj - prop center):", uv_full - proposal_center.unsqueeze(0))
    
    


def to_torch(x, device, dtype=torch.double):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    
    
# idx=0
# debug_rescaled_intrinsics(
#     K_full=to_torch(intrinsics[0], intrinsics.device),
#     K_roi=intrinsics[idx],
#     bbox=proposal_list[0][0],
#     roi_size=self.roi_size,
#     device=intrinsics.device
# )