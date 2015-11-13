from multiprocessing import Pool as ProcessPool
from multiprocessing.pool import ThreadPool

MP_POOLS = {}
MT_POOLS = {}


def get_map_fn(num_procs, use_threads=False):
  if num_procs == 1:
    return map
  if use_threads:
    if num_procs not in MT_POOLS:
      MT_POOLS[num_procs] = ThreadPool(num_procs)
    return MT_POOLS[num_procs].map
  if num_procs not in MP_POOLS:
    MP_POOLS[num_procs] = ProcessPool(num_procs)
  return MP_POOLS[num_procs].map
