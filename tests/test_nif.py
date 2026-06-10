"""Unit tests for nif.py's transform math (the code we wrote).

The PyFFI-driven block extraction is validated against real Oblivion .nif files
on the user's machine — PyFFI's reader is the mature, battle-tested path; its
*writer* can't round-trip 20.0.0.5, so we can't synthesize a fixture here.
"""
import math

from oblivion2vmf import nif

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_pyffi_available():
    # PyFFI is a hard dependency for the models pipeline; confirm the shim worked.
    assert nif.available() is True


def test_apply_identity():
    assert nif._apply(nif._identity(), (1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


def test_compose_translations():
    parent = (I3, [10.0, 0.0, 0.0], 1.0)
    child = (I3, [0.0, 5.0, 0.0], 1.0)
    w = nif._compose(parent, child)
    p = nif._apply(w, (1.0, 2.0, 3.0))
    assert tuple(round(c, 6) for c in p) == (11.0, 7.0, 3.0)


def test_compose_parent_rotation():
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    parent = (rz, [0.0, 0.0, 0.0], 1.0)
    child = (I3, [1.0, 0.0, 0.0], 1.0)
    w = nif._compose(parent, child)
    # child origin -> parent rotates child translation (1,0,0) by 90deg Z -> (0,1,0)
    p = nif._apply(w, (0.0, 0.0, 0.0))
    assert tuple(round(x, 6) for x in p) == (0.0, 1.0, 0.0)


def test_compose_parent_scale():
    parent = (I3, [0.0, 0.0, 0.0], 2.0)
    child = (I3, [1.0, 0.0, 0.0], 1.0)
    w = nif._compose(parent, child)
    # child vert (1,0,0): local -> (2,0,0); parent scale x2 -> (4,0,0)
    assert tuple(round(x, 6) for x in nif._apply(w, (1.0, 0.0, 0.0))) == (4.0, 0.0, 0.0)


def test_apply_dir_ignores_translation_and_scale():
    tf = (I3, [100.0, 50.0, 0.0], 3.0)
    assert nif._apply_dir(tf, (0.0, 0.0, 1.0)) == (0.0, 0.0, 1.0)
