import pickle

def read_model(file_path):
    with open(file_path, 'rb') as file:
            model = pickle.load(file)
    return model
