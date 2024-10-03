import os

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def check_base(base):
    path = os.path.join(base)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
