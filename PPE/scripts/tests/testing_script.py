import multiprocessing as mp
import multiprocessing.shared_memory as sm
import numpy as np

def create_arr (name):
    pass


def main():
    shared = mp.sharedctypes.RawArray(name='test', create=True, size=1000)


if __name__ == '__main__':
    main()