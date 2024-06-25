def over_write(gamma_global, gamma_fbg, gamma_instances, config_path):
    code = f"""
_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_mstrain_3x_coco/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_4',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       trans_func1='catt',
                                       trans_func2='scale_r1',
                                       trans_func3='norm_C',
                                       dis_func='l1',
                                       gamma_global = {gamma_global},
                                       gamma_fbg = {gamma_fbg},
                                       gamma_instances = {gamma_instances}
                                    )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       trans_func1='catt',
                                       trans_func2='scale_r1',
                                       trans_func3='norm_C',
                                       dis_func='l1',
                                       gamma_global = {gamma_global},
                                       gamma_fbg = {gamma_fbg},
                                       gamma_instances = {gamma_instances}
                                    )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       trans_func1='catt',
                                       trans_func2='scale_r1',
                                       trans_func3='norm_C',
                                       dis_func='l1',
                                       gamma_global = {gamma_global},
                                       gamma_fbg = {gamma_fbg},
                                       gamma_instances = {gamma_instances}
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       trans_func1='catt',
                                       trans_func2='scale_r1',
                                       trans_func3='norm_C',
                                       dis_func='l1',
                                       gamma_global = {gamma_global},
                                       gamma_fbg = {gamma_fbg},
                                       gamma_instances = {gamma_instances}
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       trans_func1='catt',
                                       trans_func2='scale_r1',
                                       trans_func3='norm_C',
                                       dis_func='l1',
                                       gamma_global = {gamma_global},
                                       gamma_fbg = {gamma_fbg},
                                       gamma_instances = {gamma_instances}
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,)
runner = dict(type='EpochBasedRunner', max_epochs=1)
"""
    with open(config_path, 'w') as file:
        file.write(code)
        file.close()

    