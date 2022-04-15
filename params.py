learning_rate = 1e-6
device = 'cuda'
batch_size = 32
weight_decay = 0
momentum = 0.9
dropout = 0.5
epochs = 100 

num_workers = 2
pin_memory = True

resume_run = False
resume_run_id = ''
visualize_preds = False
save_model_file = 'saves/model.pth.tar'
load_model_file = ''
selected_dataset = 'shape_norot'
train_data_csv = 'train.csv'
test_data_csv = 'test.csv'
optimizer = 'sgd'

S = 7
B = 2
if selected_dataset == 'voc':
    C = 20
elif selected_dataset[0:5] == 'shape':
    C = 5

architecture_size = 'mini'

enable_wandb = True

verbose = False