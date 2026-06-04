from time import perf_counter

def stopwatch(func):
    def wrapper(*args, **kwargs):
        t1_start = perf_counter()
        result = func(*args, **kwargs)
        t1_stop = perf_counter()
        
        elapsed_time = t1_stop - t1_start
        print(f"Function '{func.__name__}' executed in: {elapsed_time:.6f} seconds")
        
        return result
    return wrapper