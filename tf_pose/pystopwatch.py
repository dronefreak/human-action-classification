import time
from collections import defaultdict


class StopWatchManager:
    def __init__(self):
        self.watches = defaultdict(StopWatch)

    def get(self, name):
        return self.watches[name]

    def start(self, name):
        self.get(name).start()

    def stop(self, name):
        self.get(name).stop()

    def reset(self, name):
        self.get(name).reset()

    def get_elapsed(self, name):
        return self.get(name).get_elapsed()

    def __repr__(self):
        return '\n'.join(['%s: %.8f' % (k, v.elapsed_accumulated) for k, v in self.watches.items()])


class StopWatch:
    def __init__(self):
        self.elapsed_accumulated = 0.0
        self.started_at = time.time()

    def start(self):
        self.started_at = time.time()

    def stop(self):
        self.elapsed_accumulated += time.time() - self.started_at

    def reset(self):
        self.elapsed_accumulated = 0.0

    def get_elapsed(self):
        return self.elapsed_accumulated