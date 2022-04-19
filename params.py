
# Training parameters
learning_rate = 1e-6
batch_size = 32
weight_decay = 0
momentum = 0.9
dropout = 0.0
epochs = 100 
optimizer = 'sgd'


# Run configuration
resume_run = False
resume_run_id = ''
visualize_preds = False
save_model_file = 'saves/model.pth.tar'
load_model_file = '../input/yolo-checkpoints/04-16-2022_1.pth.tar'
selected_dataset = 'shape_norot'
train_data_csv = 'train.csv'
test_data_csv = 'test.csv'


# Model and loss configuration
S = 7
B = 2
if selected_dataset == 'voc':
    C = 20
elif selected_dataset[0:5] == 'shape':
    C = 5
architecture_size = 'mini'
losses = [
    # 'box',
    # 'class',
    'obj_conf',
    'noobj_conf'
]


# Misc
num_workers = 2
pin_memory = True
device = 'cuda'
enable_wandb = True
verbose = False