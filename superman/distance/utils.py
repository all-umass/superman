import os


def regenerate_cython(tpl_file):
  pyx_file = tpl_file[:-3]
  refresh_pyx = (not os.path.exists(pyx_file) or
                 os.path.getmtime(tpl_file) > os.path.getmtime(pyx_file))
  if refresh_pyx:
    from Cython import Tempita
    tpl = Tempita.Template.from_filename(tpl_file, encoding='utf-8')
    with open(pyx_file, 'w') as fh:
      fh.write(tpl.substitute())
