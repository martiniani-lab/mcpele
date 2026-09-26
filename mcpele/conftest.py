import importlib.util

# parallel tempering needs the optional mpi4py (pip install mcpele[mpi])
collect_ignore_glob = [] if importlib.util.find_spec("mpi4py") else ["parallel_tempering/*"]
