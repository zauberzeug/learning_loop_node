import pynvml


def get_free_memory_mb():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free = info.free / 1024 / 1024
    print(f'free     : {free} MB')
    return free
