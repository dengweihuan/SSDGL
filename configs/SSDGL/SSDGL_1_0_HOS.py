config = dict(
    model=dict(
        type='SSDGL',
        params=dict(
            in_channels=144,
            num_classes=15,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewHOSLoader',
            params=dict(
                training=True,
                num_workers=0,
                image_mat_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat',
                gt_mat_path='./GRSS2013/GRSS2013.mat',
                num_train_samples_per_class=10,
                sub_minibatch=10
            )
        ),
        test=dict(
            type='NewHOSLoader',
            params=dict(
                training=False,
                num_workers=0,
                image_mat_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat',
                gt_mat_path='./GRSS2013/GRSS2013.mat',
                num_train_samples_per_class=10,
                sub_minibatch=10
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
            base_lr=0.005,
            power=0.8,
            max_iters=1000),
    ),
    train=dict(
        forward_times=1,
        num_iters=1000,
        eval_per_epoch=True,
        summary_grads=False,
        summary_weights=False,
        eval_after_train=True,
        resume_from_last=False,
    ),
    test=dict(
        draw=dict(
            image_size=(349, 1905),
            palette=[
                0, 0, 0,
                139, 67, 45,
                0, 0, 255,
                255, 100, 0,
                0, 255, 123,
                164, 75, 155,
                101, 173, 255,
                118, 254, 172,
                60, 91, 112,
                255, 255, 0,
                255, 255, 125,
                255, 0, 255,
                100, 0, 255,
                0, 172, 254,
                0, 255, 0,
                171, 175, 80
                ]
        )
    ),
)
