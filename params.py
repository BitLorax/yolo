learning_rate = 1e-6
device = 'cpu'
batch_size = 32
weight_decay = 0
momentum = 0.9
epochs = 10

num_workers = 2
pin_memory = True

resume_run = False
resume_run_id = ''
visualize_preds = False
save_model_file = 'saves/model.pth.tar'
load_model_file = 'saves/shape/sgdm_semimini_all_10.pth.tar'
selected_dataset = 'shape'
data_csv = 'test.csv'
optimizer = 'sgd'

S = 7
B = 2
if selected_dataset == 'voc':
    C = 20
elif selected_dataset == 'shape':
    C = 5

architecture_size = 'semi-mini'