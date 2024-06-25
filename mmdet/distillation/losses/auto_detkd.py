import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchvision.transforms import ToTensor
from torch import Tensor, tensor
from einops import rearrange, reduce
from typing import List
from .operations import _DIS_FUNC, _WIT_FUNC, _TRANS_FUNC1, _TRANS_FUNC2, _TRANS_FUNC3


@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):

    def __init__(self,
                 student_channels, teacher_channels, name, 
                 global_trans1="no", global_trans2='no', global_trans3='no', global_dis='l2', gb_enable=False,
                 fbg_trans1='no', fbg_trans2='no', fbg_trans3='no', fbg_dis='l2', fbg_enable=False,
                 ins_trans1='no', ins_trans2='no', ins_trans3='no', ins_dis='l2', ins_enable=False,
                 logits_trans='no', logits_dis='no', logits_enable=False,
                 conv_trans='channel', conv_dis="l1", alpha_=0.0000005, lambda_=0.45, 
                 gamma_global=1, gamma_fbg=1, gamma_instances=1, gamma_logits=1,
                 is_conv = False
    ):
        super(FeatureLoss, self).__init__()
        self.name = name
        self.global_trans1 = global_trans1
        self.global_trans2 = global_trans2
        self.global_trans3 = global_trans3
        self.global_dis = global_dis
        self.gb_enable = gb_enable

        self.fbg_trans1 = fbg_trans1
        self.fbg_trans2 = fbg_trans2
        self.fbg_trans3 = fbg_trans3
        self.fbg_dis = fbg_dis
        self.fbg_enable = fbg_enable

        self.ins_trans1 = ins_trans1
        self.ins_trans2 = ins_trans2
        self.ins_trans3 = ins_trans3
        self.ins_dis = ins_dis
        self.ins_enable = ins_enable

        self.logits_trans = logits_trans
        self.logits_dis = logits_dis
        self.logits_enable = logits_enable

        self.gamma_global = gamma_global
        self.gamma_fbg = gamma_fbg
        self.gamma_instances = gamma_instances
        self.gamma_logits = gamma_logits
        
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.is_conv = is_conv
        self.conv_trans = conv_trans
        self.conv_dis = conv_dis

        self.align = None
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, preds_S: Tensor, preds_T: Tensor, logits_S: Tensor=None, logits_T: Tensor=None,
                regs_S: Tensor=None, regs_T: Tensor=None, img_metas: List=None, gt_bboxes: List=None):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(List): length=Bs
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        if self.is_conv:
            return self.get_conv_loss(preds_S, preds_T)

        global_loss, fbg_loss, instance_loss, logits_loss = None, None, None, None
        if self.gb_enable:
            global_loss = self.get_feat_loss(preds_S, preds_T)
        if self.fbg_enable:
            fbg_loss = self.get_fg_and_bg_loss(preds_S, preds_T, img_metas, gt_bboxes)
        if self.ins_enable:
            instance_loss = self.get_instances_loss(preds_S, preds_T, img_metas, gt_bboxes)
        if self.logits_enable:
            logits_loss = self.get_logit_loss(logits_S, logits_T)
            
        return global_loss, fbg_loss, instance_loss, logits_loss
    
    def get_conv_loss(self, preds_S, preds_T):
        dis_func = _DIS_FUNC[self.dis_func]
        trans_func = _TRANS_FUNC3[self.trans_func]
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        new_fea = trans_func(new_fea)
        preds_T = trans_func(preds_T)

        dis_loss = dis_func(new_fea, preds_T)/N

        return dis_loss
    

    def get_feat_loss(self, preds_S, preds_T):
        trans_func1 = _TRANS_FUNC1[self.global_trans1]
        trans_func2 = _TRANS_FUNC2[self.global_trans2]
        trans_func3 = _TRANS_FUNC3[self.global_trans3]
        distance_func = _DIS_FUNC[self.global_dis]
        weight = _WIT_FUNC['no']
        feat_s = trans_func3(trans_func2(trans_func1(preds_S)))
        feat_t = trans_func3(trans_func2(trans_func1(preds_T)))
        feature_loss = self.gamma_global * weight(distance_func(feat_s, feat_t))
        return feature_loss

    def get_logit_loss(self, logit_S, logit_T):
        logit_S, logit_T = logit_S.sigmoid(), logit_T.sigmoid()
        trans_func1 = _TRANS_FUNC3[self.logits_trans]
        distance_func = _DIS_FUNC[self.logits_dis]
        weight = _WIT_FUNC['no']
        logit_S_ = trans_func1(logit_S)
        logit_T_ = trans_func1(logit_T)
        logit_loss = self.gamma_logits * weight(distance_func(logit_S_, logit_T_))

        return logit_loss

    def get_fg_and_bg_loss(self, preds_S, preds_T, img_metas, gt_bboxes):
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        N, C, H, W = preds_S.shape
        Mask_fg, Mask_bg = torch.zeros((N, H, W)).to(preds_S.device), torch.ones((N, H, W)).to(preds_S.device)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])  # n 4
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())  # (n, )
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))  # 1 n

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])
        
        fg_feat_t = torch.mul(preds_T, Mask_fg.unsqueeze(1))
        bg_feat_t = torch.mul(preds_T, Mask_bg.unsqueeze(1))
        fg_feat_s = torch.mul(preds_S, Mask_fg.unsqueeze(1))
        bg_feat_s = torch.mul(preds_S, Mask_bg.unsqueeze(1))

        trans_func1 = _TRANS_FUNC1[self.fbg_trans1]
        trans_func2 = _TRANS_FUNC2[self.fbg_trans2]
        trans_func3 = _TRANS_FUNC3[self.fbg_trans3]
        distance_func = _DIS_FUNC['l1']
        weight = _WIT_FUNC['no']
        
        fg_feat_s = trans_func3(trans_func2(trans_func1(fg_feat_s))) 
        fg_feat_t = trans_func3(trans_func2(trans_func1(fg_feat_t)))
        bg_feat_s = trans_func3(trans_func2(trans_func1(bg_feat_s))) 
        bg_feat_t = trans_func3(trans_func2(trans_func1(bg_feat_t)))
        return self.gamma_fbg * weight(distance_func(fg_feat_s, fg_feat_t)), \
                self.gamma_fbg/2 * weight(distance_func(bg_feat_s, bg_feat_t))
    
    def get_instances_loss(self, preds_S, preds_T, img_metas, gt_bboxes):
        N, C, H, W = preds_S.shape
        trans_func1 = _TRANS_FUNC1[self.ins_trans1]
        trans_func2 = _TRANS_FUNC2[self.ins_trans2]
        trans_func3 = _TRANS_FUNC3[self.ins_trans3]
        distance_func = _DIS_FUNC[self.ins_dis]
        weight = _WIT_FUNC['no']

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
        
        rois_S = torch.cat(rois_S, dim=0).to(preds_S.device)
        rois_T = torch.cat(rois_T, dim=0).to(preds_S.device)
        from torchvision.ops import roi_align
        rois_feat_S = roi_align(preds_S, rois_S, (14, 14), 1.0, False)  # N C 14 14
        rois_feat_T = roi_align(preds_T, rois_T, (14, 14), 1.0, False)

        return self.gamma_instances * weight(distance_func(rois_feat_S, rois_feat_T))
    





    
    


    

