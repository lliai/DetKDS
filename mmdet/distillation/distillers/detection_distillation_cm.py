import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from mmdet.models import build_loss
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
from typing import List, Dict, Optional
from mmdet.core import multi_apply
from mmdet.distillation.losses.operations import _TRANS_FUNC1, _TRANS_FUNC2, _TRANS_FUNC3, _DIS_FUNC, _WIT_FUNC
from mmdet.distillation.losses.operations import *


@DISTILLER.register_module()
class DetectionDistiller_CM(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 rcnn_ins_trans1='no', 
                 rcnn_ins_trans2='no', 
                 rcnn_ins_trans3='no', 
                 rcnn_ins_dis='l2'): 

        super(DetectionDistiller_CM, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights()
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)
            
        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                    self.register_buffer(teacher_module,output)
            def hook_student_forward(module, input, output):
                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward

        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

        self.rcnn_ins_trans1 = rcnn_ins_trans1
        self.rcnn_ins_trans2 = rcnn_ins_trans2
        self.rcnn_ins_trans3 = rcnn_ins_trans3
        self.rcnn_ins_dis = rcnn_ins_dis
        
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, 
                      img, 
                      img_metas, 
                      **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        student_loss = self.student.forward_train(img, img_metas, **kwargs)

        with torch.no_grad():
            fea_t = self.teacher.extract_feat(img)
            tea_cls_scores, tea_reg_deltas = self.teacher.rpn_head(fea_t)
        fea_s = self.student.extract_feat(img)
        stu_cls_scores, stu_reg_deltas = self.student.rpn_head(fea_s)
                
        for i, item_loc in enumerate(self.distill_cfg):
            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                res = self.distill_losses[loss_name](
                        fea_s[i], fea_t[i], stu_cls_scores[i], tea_cls_scores[i], stu_reg_deltas[i],
                        tea_reg_deltas[i], img_metas, kwargs["gt_bboxes"])
                student_loss[f"{loss_name}_global"] = res

        return student_loss
    
    def get_instances_loss(self, preds_S, preds_T, img_metas, gt_bboxes):
        trans_func1 = _TRANS_FUNC1[self.rcnn_ins_trans1]
        trans_func2 = _TRANS_FUNC1[self.rcnn_ins_trans2]
        trans_func3 = _TRANS_FUNC1[self.rcnn_ins_trans3]
        dis_func = _DIS_FUNC[self.rcnn_ins_dis]
        weight = _WIT_FUNC['no']

        N, C, H, W = preds_S.shape

        preds_S = trans_func3(trans_func2(trans_func1(preds_S))) 
        preds_T = trans_func3(trans_func2(trans_func1(preds_T))) 

        rois_S, rois_T = [], []
        for i in range(N):
            new_boxxes = torch.ones((gt_bboxes[i].shape[0], 5))
            new_boxxes[:, 0] = torch.tensor([i for _ in range(gt_bboxes[i].shape[0])])
            new_boxxes[:, 1] = torch.floor(gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W)  # x1
            new_boxxes[:, 3] = torch.ceil(gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W)   # x2
            new_boxxes[:, 2] = torch.floor(gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H)  # y1
            new_boxxes[:, 4] = torch.ceil(gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H)   # y2

            rois_S.append(new_boxxes)
            rois_T.append(new_boxxes)
        
        rois_S = torch.cat(rois_S, dim=0).to(preds_S.device)  # (k, 5)
        rois_T = torch.cat(rois_T, dim=0).to(preds_S.device)

        preds_S, preds_T = (preds_S,), (preds_T,)
        rois_S = self.student.roi_head.bbox_roi_extractor(preds_S, rois_S)
        rois_T = self.teacher.roi_head.bbox_roi_extractor(preds_T, rois_T)
        return weight(dis_func(rois_S, rois_T))
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)