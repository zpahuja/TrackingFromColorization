"""
Timer for split (or lap) times
"""
import time


class SplitTimer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.timestamps = [('Total', time.time())]
        self.elapsed_times = {}

    def lap(self, lap_name=None):
        if lap_name is None:
            lap_name = 'Lap {}'.format(len(self.timestamps))
        self.timestamps.append((lap_name, time.time()))

    def calc(self):
        self.elapsed_times = {
            'Total': self.timestamps[-1][1] - self.timestamps[0][1]}
        self.elapsed_times.update(
            {t[0]: t[1] - self.timestamps[i][1] for i, t in enumerate(self.timestamps[1:])})
        return self.elapsed_times

    def __repr__(self):
        self.calc()
        return '\n'.join(['{}: {:.3f}'.format(key, self.elapsed_times[key]) for key, _ in self.timestamps[::-1]])
