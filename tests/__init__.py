import sys, os
path = os.path.dirname(__file__)
print(path)
#path = os.path.join(path, 'bin')
if path not in sys.path:
    sys.path.append(path)