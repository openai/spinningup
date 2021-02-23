"""Module that contains utilities that can be used with the
`openai gym package <https://github.com/openai/gym>`_.
"""

import importlib
import sys


def import_gym_env_pkg(module_name, frail=True, dry_run=False):
    """Tries to import the custom gym environment package.

    Args:
        module_name (str): The python module you want to import.
        frail (bool, optional): Throw ImportError when tensorflow can not be imported.
            Defaults to ``true``.
        dry_run (bool, optional): Do not actually import tensorflow if available.
            Defaults to ``False``.

    Raises:
        ImportError: A import error if the package could not be imported.

    Returns:
        union[tf, bool]: Custom env package if ``dry_run`` is set to ``False``.
            Returns a success bool if ``dry_run`` is set to ``True``.
    """
    module_name = module_name[0] if isinstance(module_name, list) else module_name
    try:
        if module_name in sys.modules:
            if not dry_run:
                return sys.modules[module_name]
            else:
                return True
        elif importlib.util.find_spec(module_name) is not None:
            if not dry_run:
                return importlib.import_module(module_name)
            else:
                return True
        else:
            if frail:
                raise ImportError("No module named '{}'.".format(module_name))
            return False
    except (ImportError, KeyError, AttributeError) as e:
        if ImportError:
            if not frail:
                return False
        raise e
