"""Multilingual RAG over public PDF reports, with page-level citations."""

import platform
import sys

# Intel Macs only: PyTorch (2.2.2, the last Intel-Mac build) and FAISS each bundle
# their own OpenMP runtime. If FAISS is loaded first, embedding segfaults. Loading
# PyTorch first avoids it while keeping multithreading (OMP_NUM_THREADS=1 also
# avoids the crash but made embedding 2.6x slower). Other platforms skip this.
if sys.platform == "darwin" and platform.machine() == "x86_64":
    import torch  # noqa: F401
