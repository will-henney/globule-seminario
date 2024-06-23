"""Human-readable representation of numpy data types for YAML.

Example Usage:

> import npyaml
> npyaml.register_all()
> ...
> yaml.dump(DATA_WITH_NUMPY_ARRAYS_AND_SCALARS)

This package is for when you want a transparent, easy-to-read
representation of numpy arrays and scalars in a YAML dump.  

Authors: (c) Felix Kloss 2023, William Henney 2023, 2024
Maintainer: William Henney <will@henney.org>
License: MIT

Originally based on the answer by Felix (@luator) to this
StackOverflow question:

https://stackoverflow.com/questions/75508283/dump-numpy-array-to-yaml-as-regular-list

I have added the ability to handle scalar arrays and to deal with all
the subtypes of numpy floating and integer types. This is tested with
masked arrays, which are dumped with the masked elements replaced by
"null".

Other potential corner cases, such as complex arrays, or object arrays
have not been tested, so use with care!

If you use yaml.safe_dump() instead of yaml.dump(), an exception will
be raised for any data types that do not have a human-readable YAML
representation.

One potential disadvantage is that the data will NOT be read back in
as numpy arrays (by yaml.safe_load(), for instance), but instead as
lists and python scalars. If required, you can explicitly convert the
data back to numpy arrays using np.array().

"""

import numpy as np
import yaml


def _ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    """Human-readable representation of numpy array for YAML."""
    if array.shape:
        # The simple case where the array is really an array
        return dumper.represent_list(array.tolist())
    else:
        # The complicated case where the array is a scalar
        value = array.tolist()  # yield a Python scalar.
        # We have to treat all the possible data types separately here.
        if isinstance(value, int):
            return dumper.represent_int(value)
        elif isinstance(value, float):
            return dumper.represent_float(value)
        elif isinstance(value, str):
            return dumper.represent_str(value)
        elif isinstance(value, bool):
            return dumper.represent_bool(value)
        else:
            # Just in case I have forgotten something
            raise ValueError("Unsupported type: %s" % type(value))


def _float_representer(dumper: yaml.Dumper, value: np.floating) -> yaml.Node:
    """Human-readable representation of numpy float for YAML."""
    return dumper.represent_float(float(value))


def _int_representer(dumper: yaml.Dumper, value: np.integer) -> yaml.Node:
    """Human-readable representation of numpy integer for YAML."""
    return dumper.represent_int(int(value))


def register_all():
    # Register the above functions with PyYAML: we use multi-representers
    # so that it will also work for sub classes of np.ndarray, np.floating, etc
    yaml.add_multi_representer(np.ndarray, _ndarray_representer)
    yaml.add_multi_representer(np.floating, _float_representer)
    yaml.add_multi_representer(np.integer, _int_representer)


if __name__ == "__main__":
    # Register the functions
    register_all()

    # Test it works
    a = np.array([1, 2, 3])
    print(yaml.dump({"ndarray": a}))
    aa = np.arange(9).reshape(3, 3)
    print(yaml.dump({"2-d ndarray": aa}, default_flow_style=False))
    print(yaml.dump({"inline 2-d ndarray": aa}, default_flow_style=True))
    i = np.array(1)
    print(yaml.dump({"scalar int ndarray": i}))
    x = np.array(1.0)
    print(yaml.dump({"scalar float ndarray": x}))
    i = np.int64(1)
    print(yaml.dump({"np.int64": i}))
    x = np.float64(1.0)
    print(yaml.dump({"np.float64": x}))
    am = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    print(yaml.dump({"masked array": am}))
    
