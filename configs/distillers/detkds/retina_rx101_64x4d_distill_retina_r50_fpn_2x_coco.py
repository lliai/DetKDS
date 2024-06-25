
_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller_Retina',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       ins_trans1='catt',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='l1',
                                       gamma_instances=50,
                                       ins_enable=True
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
                                       ins_trans1='catt',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='l1',
                                       gamma_instances=50,
                                       ins_enable=True
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
                                       ins_trans1='catt',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='l1',
                                       gamma_instances=50,
                                       ins_enable=True
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
                                       ins_trans1='catt',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='l1',
                                       gamma_instances=50,
                                       ins_enable=True
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
                                       ins_trans1='catt',
                                       ins_trans2='scale_r1',
                                       ins_trans3='norm_C',
                                       ins_dis='l1',
                                       gamma_instances=50,
                                       ins_enable=True
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)

