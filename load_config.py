
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    for key, val in d.items():
        if type(val) == dict:
            d[key] = dict_to_namespace(val)
    return SimpleNamespace(**d)


with open('config.yml', 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        if params['save_model_file'] is not None:
            params['save_model_filepath'] = 'saves/' + params['save_model_file']
            params['predictions_filepath'] = params['save_model_filepath'].split('.')[0] + '_predictions.npz'
        if params['load_model_file'] is not None:
            params['load_model_filepath'] = '../input/yolo-checkpoints/' + params['load_model_file']
        if params['selected_dataset']['name'] == 'voc':
            params['C'] = 20
        elif params['selected_dataset']['name'][0:5] == 'shape':
            params['C'] = 5

        p = dict_to_namespace(params)
    except yaml.YAMLError as exc:
        print(exc)
