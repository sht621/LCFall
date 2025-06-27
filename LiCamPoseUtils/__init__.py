import importlib, sys
for _sub in ('model','datasets','utils','config'):
    sys.modules[_sub]=importlib.import_module(f'{__name__}.{_sub}')

