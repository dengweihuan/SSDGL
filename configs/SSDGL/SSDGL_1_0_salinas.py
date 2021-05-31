config = dict(
    model=dict(
        type='SSDGL',
        params=dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewSalinasLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./salinas/Salinas_corrected.mat',
                gt_mat_path='./salinas/Salinas_gt.mat',
                training=True,
                sample_percent=0.01,
                batch_size=10
            )
        ),
        test=dict(
            type='NewSalinasLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./salinas/Salinas_corrected.mat',
                gt_mat_path='./salinas/Salinas_gt.mat',
                training=False,
                sample_percent=0.01,
                batch_size=10
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
            image_size=(610, 340),
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
                255, 215, 0,]
        )
    ),
)
