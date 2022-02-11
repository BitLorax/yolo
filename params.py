learning_rate = 2e-5
device = 'cpu'
batch_size = 16
weight_decay = 0
epochs = 100

num_workers = 2
pin_memory = False

load_model = True
visualize_preds = False
load_model_file = 'saves/model_100.pth.tar'
dataset = 'shape'
train_csv = 'train-100.csv'

S = 7
B = 2
if dataset == 'voc':
    C = 20
elif dataset == 'shape':
    C = 5