import pynvml


def get_free_gpu_memory():
    '''Returns the free GPU memory in MB'''
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free = info.free / 1024 / 1024
    print(f'free     : {free} MB')
    return free
