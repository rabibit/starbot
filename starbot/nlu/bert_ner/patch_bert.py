
import re
import collections
import tensorflow as tf
from bert import modeling
import inspect
import hashlib


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


origin_src = inspect.getsource(modeling.get_assignment_map_from_checkpoint)
md5 = hashlib.md5(origin_src.encode('utf8')).hexdigest()
if md5 not in {'efc6123fd597332ae8d01841bba6cf9f'}:
    raise Exception("source of bert.modeling.get_assignment_map_from_checkpoint changed, cannot patch it."
                    " md5:{}".format(md5))

modeling.get_assignment_map_from_checkpoint = get_assignment_map_from_checkpoint
