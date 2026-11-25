"""The RAM (RSS) a process holds, and its size in GB for console lines."""

import os
import subprocess
import sys

GB = 1024 ** 3


def rss(pid=None):
    """RAM (RSS) of process `pid` (default: this process) in bytes; None when unreadable. macOS
    and Linux read it from the system ps; Windows reads only this process's working set."""
    if sys.platform == "win32":
        return _own_working_set() if pid in (None, os.getpid()) else None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def gb(size):
    """A size in bytes as "2.34 GB"; "? GB" when unknown."""
    return "? GB" if size is None else f"{size / GB:.2f} GB"


def _own_working_set():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    c = Counters(cb=ctypes.sizeof(Counters))
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize if ok else None
