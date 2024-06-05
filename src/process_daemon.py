import resource
from multiprocessing import Process
from multiprocessing.pool import Pool

from torch.multiprocessing import set_sharing_strategy


class NoDaemonProcess(Process):
    """
    A process that does not start a daemon thread.
    """

    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(Pool):
    """
    A pool that does not start a daemon thread.
    """

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


# !!!!!!!!! If building the DAGs hangs, try to decrease this number !!!!!!!!!
MAXIMUM_NUMBER_OF_OPEN_FILES = 10001


def __allow_massive_parallel_processes__():
    # set_num_threads(1)
    set_sharing_strategy("file_system")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (MAXIMUM_NUMBER_OF_OPEN_FILES, hard))
