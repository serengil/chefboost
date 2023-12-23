import sys
from types import ModuleType

# pylint: disable=no-else-return


def load_module(module_name: str) -> ModuleType:
    """
    Load python module with its name
    Args:
        module_name (str): module name without .py extension
    Returns:
        module (ModuleType)
    """
    if sys.version_info >= (3, 11):
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ImportError(f"Module '{module_name}' not found")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        import imp  # pylint: disable=deprecated-module

        fp, pathname, description = imp.find_module(module_name)
        return imp.load_module(module_name, fp, pathname, description)
