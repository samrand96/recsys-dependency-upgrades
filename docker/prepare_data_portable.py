"""Run data/prepare_ml1m.py so its .npz is byte-identical to the frozen protocol-v0 artifact.

Python's zipfile stamps the creating platform into each member header (0 on Windows, 3 on
Unix). The frozen ml1m.npz was written on Windows, so on Linux the arrays are identical but
the file hash is not. Fixing that one header field reproduces SHA-256 c170bb57... exactly
(checked 14 November 2026), and the harness then accepts the data as the approved artifact.
Run from the repository root:  uv run --locked python docker/prepare_data_portable.py
"""

import runpy
import sys
import zipfile

_original_init = zipfile.ZipInfo.__init__


def _windows_platform_byte(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.create_system = 0


zipfile.ZipInfo.__init__ = _windows_platform_byte
sys.argv = ["data/prepare_ml1m.py"]
runpy.run_path("data/prepare_ml1m.py", run_name="__main__")
