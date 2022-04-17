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
            type='SmallPaviaLoader',
            params=dict(
                training=True,
                num_workers=8,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                num_train_samples_per_class=10,
                sub_minibatch=10,
                select_type='samples_per_class',                
            )
        ),
        test=dict(
            type='SmallPaviaLoader',
            params=dict(
                training=False,
                num_workers=8,
                image_mat_path='./pavia/PaviaU.mat',
                gt_mat_path='./pavia/PaviaU_gt.mat',
                num_train_samples_per_class=10,
                sub_minibatch=10,
                select_type='samples_per_class',                
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
                0, 0, 0,
                192, 192, 192,
                0, 255, 1,
                0, 255, 255,
                0, 128, 1,
                255, 0, 254,
                165, 82, 40,
                129, 0, 127,
                255, 0, 0,
                255, 255, 0, ]
        )
    ),
)
