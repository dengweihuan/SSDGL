config = dict(
    model=dict(
        type='SSDGL',
        params=dict(
            in_channels=200,
            num_classes=16,
            block_channels=(128, 192, 256,320),
            inner_dim=192,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewIndianPinesLoader',
            params=dict(
                training=True,
                num_workers=8,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                sample_percent=0.05,
                batch_size=10,
                select_type='sample_percent',
            )
        ),
        test=dict(
            type='NewIndianPinesLoader',
            params=dict(
                training=False,
                num_workers=8,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                sample_percent=0.05,
                batch_size=10,
                select_type='sample_percent',                
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.003,
            power=0.8,
            max_iters=600),

    ),
    train=dict(
        forward_times=1,
        num_iters=600,
        eval_per_epoch=True,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=False,
    ),
    test=dict(
        draw=dict(
            image_size=(145, 145),
            palette=[
                255, 0, 0,
                0, 255, 0,
                0, 0, 255,
                255, 255, 0,
                0, 255, 255,
                255, 0, 255,
                192, 192, 192,
                128, 128, 128,
                128, 0, 0,
                128, 128, 0,
                0, 128, 0,
                128, 0, 128,
                0, 128, 128,
                0, 0, 128,
                255, 165, 0,
                255, 215, 0,
            ]
        )
    ),
)
