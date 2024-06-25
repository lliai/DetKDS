
_base_ = [
    '../../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
]
# model settings
find_unused_parameters=True
alpha_=0.0000005
lambda_=0.45
distiller = dict(
    type='DetectionDistiller_CM',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_=alpha_,
                                       lambda_=lambda_,
                                       is_conv=True
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
                                       alpha_=alpha_,
                                       lambda_=lambda_,
                                       is_conv=True
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
                                       alpha_=alpha_,
                                       lambda_=lambda_,
                                       is_conv=True
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
                                       alpha_=alpha_,
                                       lambda_=lambda_,
                                       is_conv=True
                                       )
                                ]
                        ),
                   ]
    )


student_cfg = 'configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,)

