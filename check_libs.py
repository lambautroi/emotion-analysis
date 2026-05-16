import importlib
libs = ['sklearn', 'underthesea', 'datasets', 'matplotlib', 'seaborn', 'joblib']
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f'{lib}: INSTALLED')
    except ImportError:
        print(f'{lib}: MISSING')
