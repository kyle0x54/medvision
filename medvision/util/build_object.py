import sys


def build_object_from_dict(dictionary, parent=None, default_args=None):
    """ Initialize an object from dict.

    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.

    Args:
        dictionary (dict): Object types and arguments.
        parent (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.

    Returns:
        (any type): Object built from the dict.
    """
    assert isinstance(dictionary, dict) and 'type' in dictionary
    assert isinstance(default_args, dict) or default_args is None

    args = dictionary.copy()
    object_type = args.pop('type')

    if isinstance(object_type, str):
        if parent is not None:
            object_type = getattr(parent, object_type)
        else:
            object_type = sys.modules[object_type]
    elif not isinstance(object_type, type):
        raise TypeError('unsupported type {}'.format(type(object_type)))

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return object_type(**args)
