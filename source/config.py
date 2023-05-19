"""
    contains all configurations, either manually set by the user or by arguments
"""


def _print_self(self):
    print([(arg, getattr(self, arg)) for arg in dir(self) if not arg.startswith("_")])


def _get_as_dict(self):
    me_as_dict = {}
    for arg in dir(self):
        if not arg.startswith("_"):
            me_as_dict[arg] = getattr(self, arg)
    return me_as_dict

