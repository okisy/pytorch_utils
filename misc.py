import torch
import argparse

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def tsr2numpy(tsr):
    return tsr.to('cpu').detach().numpy().copy()

class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='debug')
    parser.add_argument('--config', type=str, default='../configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.load(fin))
    print(args.__dict__)
