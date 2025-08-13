# For relative imports to work in Python 3.6
import os
import sys


sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # noqa: E401, E702, I001
