import os
import pickle


def save_stub(stubs, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump(stubs, f)


def read_stub(read_from_stub, stub_path):
    if read_from_stub and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None
