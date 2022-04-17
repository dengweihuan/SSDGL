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
            type='SmallIndianPinesLoader',
            params=dict(
                num_workers=0,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                training=True,
                num_train_samples_per_class=10,
                sub_minibatch=10,
                select_type='samples_per_class',
            )
        ),
        test=dict(
            type='SmallIndianPinesLoader',
            params=dict(

                num_workers=4,
                image_mat_path='./IndianPines/Indian_pines_corrected.mat',
                gt_mat_path='./IndianPines/Indian_pines_gt.mat',
                training=False,
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
            max_iters=1000),
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
