_base_ = [
    '../../gfl/gfl_r50_fpn_mstrain_2x_coco.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_GFL',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       global_trans1='catt',
                                       global_trans2='scale_r2',
                                       global_trans3='norm_N',
                                       global_dis='l1',
                                       gb_enable=True,
                                       gamma_global=10, 
                                       )
                                ]
                        ),
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
                                       gb_enable=True,
                                       gamma_global=10, 
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
                                       gb_enable=True,
                                       gamma_global=10, 
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
                                       gb_enable=True,
                                       gamma_global=10, 
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
                                       gb_enable=True,
                                       gamma_global=10, 
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/gfl/gfl_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=24)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)