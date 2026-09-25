import logging as _mcpele_logging
import sys as _mcpele_sys

logger = _mcpele_logging.getLogger("mcpele")
global_handler = _mcpele_logging.StreamHandler(_mcpele_sys.stdout)
logger.addHandler(global_handler)
logger.setLevel(_mcpele_logging.DEBUG)


def get_include():
    """Directory with mcpele's C++ sources and headers (`mcpele/*.h`), for
    building extensions against mcpele. Works for installed packages and for
    in-place builds of a source checkout."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    installed = os.path.join(here, "source")
    return installed if os.path.isdir(installed) else os.path.join(os.path.dirname(here), "source")
