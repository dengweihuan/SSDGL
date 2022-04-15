config = dict(
    model=dict(
        type='SSDGL',
        params=dict(
            in_channels=103,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),
    data=dict(
        train=dict(
            type='NewPaviaLoader',
            params=dict(
                training=True,
                num_workers=8,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                sample_percent=0.01,
                batch_size=20,
                select_type='sample_percent',                
            )
        ),
        test=dict(
            type='NewPaviaLoader',
            params=dict(
                training=False,
                num_workers=8,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                sample_percent=0.01,
                batch_size=20,
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
            base_lr=0.001,
            power=0.9,
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
                128, 0, 0, ]
        )
    ),
)
