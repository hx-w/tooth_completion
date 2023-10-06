# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from collections import OrderedDict

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)
