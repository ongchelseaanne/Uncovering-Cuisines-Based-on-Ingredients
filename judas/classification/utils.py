import time, sys

def progressbar(it, prefix="", text="", size=40, file=sys.stdout):
    count = len(it)
    def show(j,):
        time_msg = " | Execution time: {:.2f}s\r".format(time.time() - start_time)
        x = int(size*j/count)
        msg = "%s[%s%s] %s %i/%i" % (prefix, "#"*x, "."*(size-x), text, j, count)
        file.write(msg + time_msg)
        file.flush()    
    start_time = time.time()
    show(0)
    for i, item in enumerate(it):
        yield item
        time_msg = "Execution time: {:.2f}s".format(time.time() - start_time)
        show(i+1)
    file.write("\n")
    file.flush()
