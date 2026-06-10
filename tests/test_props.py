import math

from oblivion2vmf.props import oblivion_matrix, placement_origin, refr_to_angles


def test_identity_rotation():
    p, y, r = refr_to_angles(0.0, 0.0, 0.0)
    assert abs(p) < 1e-6 and abs(y) < 1e-6 and abs(r) < 1e-6


def test_matrix_orthonormal():
    m = oblivion_matrix(0.3, -0.7, 1.1)
    # columns unit length and mutually orthogonal
    cols = [[m[r][c] for r in range(3)] for c in range(3)]
    for c in cols:
        assert abs(math.sqrt(sum(x * x for x in c)) - 1.0) < 1e-9
    dot = sum(cols[0][i] * cols[1][i] for i in range(3))
    assert abs(dot) < 1e-9


def test_angles_roundtrip_via_matrix():
    # refr_to_angles must be the exact inverse of Source's AngleMatrix; rebuild
    # the Source matrix from the returned angles and compare to oblivion_matrix.
    rX, rY, rZ = 0.2, -0.5, 0.9
    pitch, yaw, roll = refr_to_angles(rX, rY, rZ)
    src = _angle_matrix(pitch, yaw, roll)
    obl = oblivion_matrix(rX, rY, rZ)
    for i in range(3):
        for j in range(3):
            assert abs(src[i][j] - obl[i][j]) < 1e-6


def test_z_only_rotation_is_pure_yaw():
    # rotation only about Z -> only yaw should be non-zero
    pitch, yaw, roll = refr_to_angles(0.0, 0.0, math.radians(30))
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6
    assert abs(abs(yaw) - 30.0) < 1e-4


def test_yaw_offset_applied():
    # identity rotation + -90 offset -> yaw -90
    p, y, r = refr_to_angles(0.0, 0.0, 0.0, yaw_offset=-90.0)
    assert abs(p) < 1e-6 and abs(r) < 1e-6 and abs(y + 90.0) < 1e-6
    # offset stacks on the computed yaw
    p2, y2, r2 = refr_to_angles(0.0, 0.0, math.radians(30), yaw_offset=-90.0)
    base = refr_to_angles(0.0, 0.0, math.radians(30))[1]
    assert abs(y2 - (base - 90.0)) < 1e-6


def test_yaw_offset_is_matrix_rotation_for_tilted():
    # For a TILTED prop the offset must be a local-Z matrix rotation (R @ Rz),
    # not a scalar add to yaw. Verify the returned angles round-trip to
    # oblivion_matrix @ Rz(offset).
    from oblivion2vmf.props import _matmul3, _rot_z
    rX, rY, rZ, off = 0.6, -0.3, 1.0, -90.0
    pitch, yaw, roll = refr_to_angles(rX, rY, rZ, yaw_offset=off)
    expected = _matmul3(oblivion_matrix(rX, rY, rZ), _rot_z(math.radians(off)))
    got = _angle_matrix(pitch, yaw, roll)
    for i in range(3):
        for j in range(3):
            assert abs(got[i][j] - expected[i][j]) < 1e-6


def test_placement_origin_recenter_scale():
    o = placement_origin((100.0, 200.0, 300.0), (50.0, 50.0, 0.0), 0.5)
    assert o == ((100 - 50) * 0.5, (200 - 50) * 0.5, 300 * 0.5)


def _angle_matrix(pitch, yaw, roll):
    # Valve AngleMatrix (degrees) -> 3x3, column-major basis (verbatim formula)
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    m = [[0.0] * 3 for _ in range(3)]
    m[0][0] = cp * cy
    m[1][0] = cp * sy
    m[2][0] = -sp
    m[0][1] = sp * sr * cy - cr * sy
    m[1][1] = sp * sr * sy + cr * cy
    m[2][1] = sr * cp
    m[0][2] = sp * cr * cy + sr * sy
    m[1][2] = sp * cr * sy - sr * cy
    m[2][2] = cr * cp
    return m
