"""Oblivion REFR placement -> Source prop_static math.

Rotation: Oblivion stores (rX, rY, rZ) in RADIANS as *negative* rotations applied
Z then Y then X (Z innermost), Z-up right-handed. Source `prop_static` "angles" is
"pitch yaw roll" in degrees (pitch=Y, yaw=Z, roll=X), composed Rz*Ry*Rx. The two
Euler orders are reversed, so we build the Oblivion orientation matrix and then
decompose it with Valve's own MatrixAngles formula (exact round-trip with Source's
AngleMatrix).
"""
from __future__ import annotations

import math

ANGLE_SIGN = -1.0   # Oblivion angles are negative rotations; flip to +1.0 if mirrored


def _matmul3(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def oblivion_matrix(rX, rY, rZ, sign=ANGLE_SIGN):
    """Orientation matrix R such that world_dir = R @ local_dir, columns = the
    object's local +X/+Y/+Z axes in world space."""
    a, b, c = sign * rX, sign * rY, sign * rZ
    # Z applied first (innermost) -> Y -> X (outermost): R = Rx @ Ry @ Rz
    return _matmul3(_rot_x(a), _matmul3(_rot_y(b), _rot_z(c)))


def refr_matrix(rX, rY, rZ, sign=ANGLE_SIGN, yaw_offset=0.0):
    """The 3x3 world orientation a placed prop ends up with — i.e. the matrix the
    engine builds from :func:`refr_to_angles`. Use it to bake prop geometry into
    world space (world_vert = origin + R @ local_vert)."""
    m = oblivion_matrix(rX, rY, rZ, sign)
    if yaw_offset:
        m = _matmul3(m, _rot_z(math.radians(yaw_offset)))
    return m


def refr_to_angles(rX, rY, rZ, sign=ANGLE_SIGN, yaw_offset=0.0):
    """Convert an Oblivion REFR rotation (radians) to Source (pitch, yaw, roll)
    degrees, via Valve's MatrixAngles decomposition.

    ``yaw_offset`` (degrees) corrects the constant Oblivion-vs-Source facing
    difference. It is applied as a rotation about the model's local up axis
    *before* decomposition (R' = R @ Rz(offset)), NOT added to the decomposed yaw
    — those are identical for upright props but only the matrix form is correct
    once a prop has real pitch/roll (tilted rocks, angled signs, fallen logs)."""
    m = oblivion_matrix(rX, rY, rZ, sign)
    if yaw_offset:
        m = _matmul3(m, _rot_z(math.radians(yaw_offset)))
    # Column vectors (Source basis convention)
    fwd = (m[0][0], m[1][0], m[2][0])   # local +X (forward)
    left = (m[0][1], m[1][1], m[2][1])  # local +Y (left)
    up_z = m[2][2]
    xy = math.hypot(fwd[0], fwd[1])
    if xy > 1e-3:
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        pitch = math.degrees(math.atan2(-fwd[2], xy))
        roll = math.degrees(math.atan2(left[2], up_z))
    else:                                # gimbal lock (pointing near ±Z)
        yaw = math.degrees(math.atan2(-left[0], left[1]))
        pitch = math.degrees(math.atan2(-fwd[2], xy))
        roll = 0.0
    return pitch, yaw, roll


def placement_origin(pos, offsets, scale):
    """World position -> recentered, scaled Hammer origin (x, y, z)."""
    ox, oy, oz = pos
    xo, yo, zo = offsets
    return ((ox - xo) * scale, (oy - yo) * scale, (oz - zo) * scale)
