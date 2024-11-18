import time


class Timer:
    """Timer class.

    Example:
    >>> import time
    >>> import medvision as mv
    >>> with mv.Timer(description='it takes {:.2f} seconds.'):
    >>>     time.sleep(1)
    it takes 1.00 seconds.
    """

    def __init__(self, description=None):
        self.description = description if description else "{:.2f}"

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t_end = time.time()
        print(self.description.format(self.t_end - self.t_start))


if __name__ == "__main__":
    with Timer(description="function takes {:.4f} seconds."):
        time.sleep(1)
