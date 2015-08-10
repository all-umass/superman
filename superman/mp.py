import multiprocessing

MP_POOLS = {}


def get_mp_pool(num_procs):
  if num_procs not in MP_POOLS:
    MP_POOLS[num_procs] = multiprocessing.Pool(num_procs)
  return MP_POOLS[num_procs]
