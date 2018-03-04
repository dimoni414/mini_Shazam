from Profiler import Profiler
import multiprocessing as mp
import os
import wave
from array import array
import time
import numpy as np
import math
import matplotlib.pyplot as plt

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}




class Worker(mp.Process):
    def __init__(self, jobs, divide_by):
        self.jobs = jobs
        self.divide_by = divide_by
        mp.Process.__init__(self)
        self.file_name = None
        self.stream = None
        self.nchannels = None
        self.samp_width = None
        self.framerate = None
        self.nframes = None
        self.comptype = None
        self.compname = None
        self.int_width = None
        self.nframes_to_read = None

    def _reinitialize(self, file_name):
        self.file_name = file_name
        self.stream = wave.open(os.path.join('Audio', file_name), mode="r")
        (self.nchannels
         , self.samp_width
         , self.framerate
         , self.nframes
         , self.comptype
         , self.compname) = self.stream.getparams()
        self.int_width = types[self.samp_width]
        self.nframes_to_read = self.nframes // self.divide_by

    def run(self):
        while True:
            self._reinitialize(self.jobs.get())
            with Profiler(self.file_name) as p:
                print("%s %s" % (self.name, self.file_name))
                self._work()
                self.jobs.task_done()

    def _work(self):
        data = self.stream.readframes(self.nframes_to_read)
        mono_samples = array('f')
        while data:
            mono_samples.extend(self._to_mono(data, self.samp_width, self.nchannels))
            data = self.stream.readframes(self.nframes_to_read)
        downsample_result = self._downsample(mono_samples)
        self.make_spectrogram(downsample_result)
        # self._convert_to_wave_file(downsample_result)

    def _to_mono(self, frames, samp_width, nchannels):
        samples = np.fromstring(frames, dtype=self.int_width)
        channel = []
        for i in range(nchannels):
            channel.append([])
        if nchannels == 1:
            return samples
        else:
            for i in range(nchannels):
                channel[i] = samples[i::nchannels]
        result_mono_channel = np.array(channel[0])
        for i in range(1, nchannels):
            result_mono_channel += channel[i]
        return result_mono_channel / nchannels

    def _downsample(self, samples):
        REDUCE_INDEX = 4
        downsample_result = np.array([sum(samples[i:i + REDUCE_INDEX]) for i in range(0, len(samples), REDUCE_INDEX)])
        return downsample_result / REDUCE_INDEX

    # TODO вероятно надо запоминать в группах ещё и значение наисильнейшей частоты
    def make_spectrogram(self, samples):
        WINDOW_SIZE = 1024
        window = np.hamming(WINDOW_SIZE)

        bands_arr = np.zeros((math.ceil(len(samples)/WINDOW_SIZE),6),dtype='f')
        band_index=0
        for nsamples in np.array([samples[i:i + WINDOW_SIZE] for i in range(0, len(samples), WINDOW_SIZE)]):
            if len(nsamples) == WINDOW_SIZE:
                bins = np.abs(np.fft.rfft(nsamples * window))  # Преобразование Фурье

            grouped_bins = np.array( [bins[0:10]
                                    , bins[11:20]
                                    , bins[21:40]
                                    , bins[41:80]
                                    , bins[81:160]
                                    , bins[161:len(bins)]]) # Делю на группы very low, low, low-mid, mid....

            for gr_index in range(len(grouped_bins)):       # Нахожу в каждой группе сильнейшие Бины
                strongest_bin = max(grouped_bins[gr_index])
                bands_arr[band_index][gr_index] = strongest_bin
            band_index += 1

        bands_arr = self._sift_below_average_freq(bands_arr, len(grouped_bins)) # Оставляю только те бины, которые больше среднего в песне
        pc = plt.pcolor(np.transpose(bands_arr))  # метод псевдографики pcolor
        plt.colorbar(pc)

        plt.title('Simple pcolor plot')
        plt.xlabel("Время")
        plt.ylabel("Частоты")
        plt.grid(True)
        plt.show()
        # plt.bar(np.fft.rfftfreq(WINDOW_SIZE, 1. / 11025), np.abs(bins) / WINDOW_SIZE)
        # plt.xlabel("Частоты")
        # plt.ylabel("Амплитуда")
        # plt.grid(True)
        # plt.show( )

    def _sift_below_average_freq(self, bands_arr, groups_count):
        FILTRATION_COEFFICIENT = 2
        for gr_index in range(groups_count):
            cur_average = np.average(bands_arr[:,gr_index]) * FILTRATION_COEFFICIENT
            for i in range(len(bands_arr)):
                if bands_arr[i][gr_index]<cur_average:
                    bands_arr[i][gr_index]=None
        return bands_arr

    # def _select_band(self, bin):
    #     if bin < 10:
    #         return 0
    #     elif bin < 20:
    #         return 1
    #     elif bin < 40:
    #         return 2
    #     elif bin < 80:
    #         return 3
    #     elif bin < 160:
    #         return 4
    #     elif bin < 511:
    #         return 5

    def _convert_to_wave_file(self, samples):
        with wave.open("Result Audio\\Test_" + self.file_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.samp_width)
            wf.setframerate(self.framerate // 4)
            temp = np.array(samples, dtype=self.int_width).tobytes()
            wf.writeframes(temp)


