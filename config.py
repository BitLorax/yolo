
# Learning parameters
epochs = 300
batch_size = 32
optimizer = 'adam'
learning_rate = 1e-5
momentum = 0.9
weight_decay = 0

# Run configuration
resume_run = False
resume_run_id = None
visualize_preds = False
save_model_file = 'model.pth.tar'
load_model_file = 'model.pth.tar'
save_preds_file = 'preds.npz'
load_preds_file = 'preds.npz'

# Model and loss configuration
S = 7
B = 2
C = 5
architecture_size = 'mini_dense'
dropout = 0.5
losses = ['box', 'class', 'obj_conf', 'noobj_conf']

# Misc
num_workers = 2
pin_memory = True
device = 'cuda'
enable_wandb = True

# Architecture hyperparameters
# (size, padding, stride, padding)
# M for max pool
# [(...)..., repeats]

conv_architectures = {
    'full': [
        (7, 64, 2, 3),
        'M',
        (3, 192, 1, 1),
        'M',
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        'M',
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        'M',
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ],
    'mini': [
        (7, 64, 2, 3),
        'M',
        (3, 192, 1, 1),
        'M',
        (3, 256, 1, 1),
        (3, 512, 1, 1),
        'M',
        (3, 512, 1, 1),
        'M',
        (3, 512, 1, 1),
        (3, 512, 2, 1),
        (3, 512, 1, 1),
        (3, 1024, 1, 1)
    ],
    'semi_mini': [
        (7, 64, 2, 3),
        'M',
        (3, 192, 1, 1),
        'M',
        (3, 256, 1, 1),
        (3, 512, 1, 1),
        'M',
        (3, 512, 1, 1),
        'M',
        (3, 512, 1, 1),
        (3, 512, 2, 1),
        (3, 512, 1, 1),
        (3, 1024, 1, 1)
    ],
    'mini_dense': [
        (7, 64, 2, 3),
        'M',
        (3, 192, 1, 1),
        'M',
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        'M',
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        'M',
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
    ]
}
dense_sizes = {
    'full': 4096,
    'mini': 512,
    'semi_mini': 1024,
    'mini_dense': 512,
}
conv_architecture = conv_architectures[architecture_size]
dense_size = dense_sizes[architecture_size]