# -*- mode: python ; coding: utf-8 -*-
import os
import sys

site_packages = os.path.join(
    os.path.dirname(sys.executable), "Lib", "site-packages"
)

a = Analysis(
    ['censor_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        # NudeNet ONNX model
        (os.path.join(site_packages, 'nudenet', '320n.onnx'), 'nudenet'),
    ],
    hiddenimports=[
        'nudenet',
        'nudenet.nudenet',
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        'cv2',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude everything we don't need
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras',
        'scipy', 'pandas', 'matplotlib',
        'IPython', 'jupyter', 'notebook', 'nbformat',
        'PIL', 'pillow',
        'h5py', 'boto3', 'botocore',
        'pyarrow', 'duckdb', 'sqlalchemy',
        'pytest', 'black', 'pygments',
        'numba', 'llvmlite',
        'lxml', 'openpyxl',
        'cryptography', 'nacl', 'bcrypt',
        'zmq', 'tornado',
        'jedi', 'parso',
        'setuptools', 'pkg_resources',
        'jsonschema', 'jsonschema_specifications',
        'rich', 'anyio',
        'pydantic', 'pydantic_core',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoCensor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='AutoCensor',
)
