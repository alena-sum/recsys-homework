"""Execute with the current Python, without installing a global kernel spec."""
from pathlib import Path
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parent
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache' / 'matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(root / '.cache' / 'ipython'))
(root / '.cache').mkdir(exist_ok=True)


if __name__ == '__main__':
    path = root / 'experiments.ipynb'
    nb = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix='recsys-kernel-') as tmp:
        import json
        spec = Path(tmp) / 'recsys'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps({
            'argv':[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name':'RecSys homework', 'language':'python'}))
        ksm = KernelSpecManager(kernel_dirs=[tmp])
        km = KernelManager(kernel_name='recsys', kernel_spec_manager=ksm)
        client = NotebookClient(nb, km=km, timeout=1800,
            resources={'metadata':{'path':str(root)}})
        def progress(cell, cell_index, **kwargs):
            print(f'Cell {cell_index + 1}/{len(nb.cells)}', flush=True)
        client.on_cell_start = progress
        try:
            client.execute()
        finally:
            nbformat.write(nb, path)
    print('Saved executed experiments.ipynb', flush=True)
