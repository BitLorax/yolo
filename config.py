
# Learning parameters
epochs = 1
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
enable_wandb = False