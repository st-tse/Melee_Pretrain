import errno
import os
import sys

def createdirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def log(str, f):
    print(str, file=sys.stderr)
    print(str, file=f)