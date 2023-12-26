import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    """
    NoDaemonProcess class for recursive parallel runs
    """
    def _get_daemon(self):
        # make 'daemon' attribute always return False
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonContext(type(multiprocessing.get_context())):
    """
    NoDaemonContext class for recursive parallel runs
    """
    # pylint: disable=too-few-public-methods
    Process = NoDaemonProcess


class CustomPool(multiprocessing.pool.Pool):
    """
    MyPool class for recursive parallel runs
    """
    # pylint: disable=too-few-public-methods, abstract-method, super-with-arguments
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(CustomPool, self).__init__(*args, **kwargs)
