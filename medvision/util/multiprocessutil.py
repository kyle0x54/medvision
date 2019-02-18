import multiprocessing
from tqdm import tqdm


def tqdm_imap(func, args, n_processes=None):
    """ Parallel processing with a progress bar.

    Run a function multiple times (with different inputs) utilizing parallel
    computing. During the parallel processing a progress bar is shown to
    check the processing progress.

    Args:
        func (function obj): the function to be called.
        args (list): the function arguments.
        n_processes (int): number of processes to be utilized.

    Return:
        (list): the results for different inputs.
    """
    p = multiprocessing.Pool(n_processes)

    result = []
    for res in tqdm(p.imap(func, args), total=len(args)):
        result.append(res)

    p.close()
    p.join()

    return result


# Multiprocessing cannot be tested in pytest.
# We can test it here instead.
if __name__ == '__main__':
    def add(a, b, c):
        return a + b + c

    first = [str(i) for i in range(100)]
    second = '_m'
    third = '.png'

    # single processor version
    result_sp = [add(i, second, third) for i in first]

    # multi-processors version 1
    def add_wrapper(xyz):
        return add(*xyz)
    in_para_wrapper = [(i, second, third) for i in first]
    result_mp = tqdm_imap(add_wrapper, in_para_wrapper)
    assert result_sp == result_mp

    # multi-processors version 2
    from functools import partial
    partial_func = partial(add, second=second, third=third)
    result_mp = tqdm_imap(partial_func, first)
    assert result_sp == result_mp
