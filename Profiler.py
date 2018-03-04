import time


class Profiler(object):
    def __init__(self, name_func="Затраченное время"):
        self.name_func = name_func

    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("{s}: {t:.3f} sec".format(s=self.name_func, t=time.time() - self._startTime))
