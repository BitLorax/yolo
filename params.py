learning_rate = 1e-6
device = 'cuda'
batch_size = 32
weight_decay = 0
momentum = 0.9
epochs = 3

num_workers = 2
pin_memory = False

load_model = False
visualize_preds = False
load_model_file = 'saves/model.pth.tar'
selected_dataset = 'shape'
data_csv = 'train.csv'
optimizer = 'sgd'

S = 7
B = 2
if selected_dataset == 'voc':
    C = 20
elif selected_dataset == 'shape':
    C = 5

architecture_size = 'mini'