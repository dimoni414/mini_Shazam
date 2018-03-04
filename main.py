import SpectrogramMaker as sm
from Profiler import Profiler
import multiprocessing as mp

if __name__ == '__main__':

    num_worker_process = 1
    jobs = ["50_1.wav"]
    # ,"50_2.wav","50_3.wav","50_11.wav","50_21.wav","50_31.wav","splin.wav"]


    j_queue = mp.JoinableQueue()

    for job in jobs:
        j_queue.put(job)
    with Profiler("Общее время работы:"):
        for i in range(num_worker_process):
            process = sm.Worker(j_queue, 64)
            process.daemon = True
            process.start()
    j_queue.join()
