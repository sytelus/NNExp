from functools import wraps
import gc
import timeit

def MeasureTime(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if gcold:
                gc.enable()
            print('Function "{}": {}s'.format(f.__name__, elapsed))
        return result
    return _wrapper

class MeasureBlockTime:
    def __init__(self,name="(block)", no_print = False, disable_gc = True, format_str=":.2f"):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
        self.format_str = format_str
    def __enter__(self):
        if self.disable_gc:
            self.gcold = gc.isenabled()
            gc.disable()
        self.start_time = timeit.default_timer()
        return self
    def __exit__(self,ty,val,tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        if not self.no_print:
            print(('{}: {' + self.format_str + '}s').format(self.name, self.elapsed))
        return False #re-raise any exceptions

def getTime():
    return timeit.default_timer()
def getElapsedTime(start_time):
    return timeit.default_timer() - start_time

