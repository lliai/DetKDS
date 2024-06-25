
_base_ = [
    '../../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_FR',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',
    init_student = True,
    distill_cfg = [
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='catt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_N',
                                       global_dis='l1',
                                       fbg_trans1="mask",
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       ins_trans1='mask',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='pear',
                                       logits_trans='softmax_C',
                                       logits_dis='pear',
                                       fbg_enable=True,
                                       gb_enable=True,
                                       ins_enable=True,
                                       logits_enable=True,
                                       gamma_global=2, 
                                       gamma_fbg=2,
                                       gamma_ins=0.1,
                                       gamma_logits=0.1
                                    )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='catt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_N',
                                       global_dis='l1',
                                       fbg_trans1="mask",
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       ins_trans1='mask',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='pear',
                                       logits_trans='softmax_C',
                                       logits_dis='pear',
                                       fbg_enable=True,
                                       gb_enable=True,
                                       ins_enable=True,
                                       logits_enable=True,
                                       gamma_global=2, 
                                       gamma_fbg=2,
                                       gamma_ins=0.1,
                                       gamma_logits=0.1
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='catt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_N',
                                       global_dis='l1',
                                       fbg_trans1="mask",
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       ins_trans1='mask',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='pear',
                                       logits_trans='softmax_C',
                                       logits_dis='pear',
                                       fbg_enable=True,
                                       gb_enable=True,
                                       ins_enable=True,
                                       logits_enable=True,
                                       gamma_global=2, 
                                       gamma_fbg=2,
                                       gamma_ins=0.1,
                                       gamma_logits=0.1
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='catt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_N',
                                       global_dis='l1',
                                       fbg_trans1="mask",
                                       fbg_trans2='no',
                                       fbg_trans3='norm_C',
                                       fbg_dis='l2',
                                       ins_trans1='mask',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='pear',
                                       logits_trans='softmax_C',
                                       logits_dis='pear',
                                       fbg_enable=True,
                                       gb_enable=True,
                                       ins_enable=True,
                                       logits_enable=True,
                                       gamma_global=2, 
                                       gamma_fbg=2,
                                       gamma_ins=0.1,
                                       gamma_logits=0.1
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
