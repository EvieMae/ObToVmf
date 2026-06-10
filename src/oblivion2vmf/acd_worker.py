"""Subprocess entry point for isolated CoACD decomposition.

Run as ``python -m oblivion2vmf.acd_worker <in.pkl> <out.pkl>``. Reads pickled
(submeshes, kwargs), runs the in-process decomposition, and writes the pickled
result. Running CoACD here (a separate process) means a native crash on a
degenerate mesh kills only this worker, not the whole build — the parent sees a
non-zero exit and falls back to non-solid collision for that prop.
"""
import pickle
import sys


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        submeshes, kwargs = pickle.load(f)
    from .acd import decompose
    parts = decompose(submeshes, **kwargs)
    with open(out_path, "wb") as f:
        pickle.dump(parts, f)


if __name__ == "__main__":
    main()
