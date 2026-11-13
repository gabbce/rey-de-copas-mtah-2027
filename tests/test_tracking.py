import numpy as np

from possession import features
from possession.tracking import AWAY, HOME, Tracking, load_tracking, read_tracking

PITCH = (100.0, 50.0)

META = """<main><Metadata>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T1"/>
</Players></Metadata></main>"""

RAW = (
    "1:0.5,0.5,1.2;NaN,NaN,NaN;0.1,0.2,0.0:0.5,0.25,3.0\n"
    "2:0.5,0.5,1.2;0.6,0.4,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n"
)


def write_bundle(tmp_path):
    meta, raw = tmp_path / "atd.xml", tmp_path / "atd.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    return meta, raw


def test_read_tracking_scales_to_metres_and_aligns_frame_ids(tmp_path):
    tracking = read_tracking(*write_bundle(tmp_path), 3, PITCH)
    assert tracking.team.tolist() == [HOME, AWAY, HOME]
    assert np.allclose(tracking.ball[0], [50.0, 12.5])
    assert np.isnan(tracking.ball[1]).all()
    assert np.allclose(tracking.players[1, 1], [60.0, 20.0])
    assert np.isnan(tracking.players[0, 1]).all()
    # frame_id 1 is video frame 0; a frame the file never mentions stays empty
    assert np.isnan(tracking.players[2]).all() and np.isnan(tracking.ball[2]).all()


def test_load_tracking_caches_and_returns_the_same_arrays(tmp_path):
    meta, raw = write_bundle(tmp_path)
    first = load_tracking(meta, raw, 3, PITCH, tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("atd-*.npz"))) == 1
    second = load_tracking(meta, raw, 3, PITCH, tmp_path / "cache")
    assert np.array_equal(first.ball, second.ball, equal_nan=True)
    assert np.array_equal(first.players, second.players, equal_nan=True)


def test_features_measure_distance_to_each_team():
    nan = np.nan
    tracking = Tracking(
        ball=np.array([[10.0, 10.0], [nan, nan], [-1.0, 5.0]], dtype=np.float32),
        players=np.array([
            [[11.0, 10.0], [10.0, 13.0], [nan, nan]],
            [[11.0, 10.0], [10.0, 13.0], [nan, nan]],
            [[0.0, 5.0], [nan, nan], [nan, nan]],
        ], dtype=np.float32),
        team=np.array([HOME, AWAY, HOME], dtype=np.int8),
    )
    f = features.compute(tracking, fps=25, pitch=PITCH)
    assert f.ball_seen.tolist() == [True, False, True]
    assert np.allclose([f.d_home[0], f.d_away[0]], [1.0, 3.0])
    assert np.isnan(f.d_home[1]) and np.isnan(f.d_away[1])
    assert np.isnan(f.d_away[2])  # no away player tracked on that frame
    assert f.ball_out.tolist() == [False, False, True]


def test_player_speed_ignores_identity_switch_jumps():
    players = np.zeros((3, 3, 2), dtype=np.float32)
    players[1, :, 0] = [0.04, 0.04, 30.0]  # one slot teleports 30 m in a frame
    tracking = Tracking(np.zeros((3, 2), dtype=np.float32), players,
                        np.array([HOME, AWAY, HOME], dtype=np.int8))
    f = features.compute(tracking, fps=25, pitch=PITCH)
    assert np.isnan(f.player_speed[0])
    assert np.isclose(f.player_speed[1], 1.0)


def test_ball_speed_is_measured_over_five_frames_and_skips_gaps():
    ball = np.zeros((9, 2), dtype=np.float32)
    ball[:, 0] = np.arange(9) * 0.4  # 0.4 m per frame = 10 m/s at 25 fps
    ball[7] = np.nan
    tracking = Tracking(ball, np.zeros((9, 1, 2), dtype=np.float32),
                        np.array([HOME], dtype=np.int8))
    speed = features.compute(tracking, fps=25, pitch=PITCH).ball_speed
    assert np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all()
    assert np.allclose(speed[2:5], 10.0)
    assert np.isnan(speed[5])  # its window ends on the missing frame
    assert np.isclose(speed[6], 10.0)  # a gap strictly inside the window does not matter


def test_the_cache_is_not_reused_when_the_pitch_or_metadata_change(tmp_path):
    meta, raw = write_bundle(tmp_path)
    cache = tmp_path / "cache"
    load_tracking(meta, raw, 3, PITCH, cache)
    wider = load_tracking(meta, raw, 3, (200.0, 50.0), cache)
    assert np.allclose(wider.ball[0], [100.0, 12.5])  # re-parsed with the new pitch
    meta.write_text(META.replace('<Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away'
                                 '</Name></Team>', '<Team id="T2"><Name>Away</Name></Team>'
                                 '<Team id="T1"><Name>Home</Name></Team>'))
    swapped = load_tracking(meta, raw, 3, PITCH, cache)
    assert swapped.team.tolist() == [AWAY, HOME, AWAY]


def test_rolling_means_ignore_gaps_and_can_exclude_the_frame_itself():
    from possession.features import _rolling_mean
    values = np.array([1.0, np.nan, 3.0, 5.0])
    assert np.allclose(_rolling_mean(values, 1, 1), [1.0, 2.0, 4.0, 4.0])
    before = _rolling_mean(values, 2, -1)
    assert np.isnan(before[0]) and np.allclose(before[1:], [1.0, 1.0, 3.0])


def test_spread_is_the_mean_distance_to_the_team_centroid():
    players = np.full((1, 4, 2), np.nan, dtype=np.float32)
    players[0, 0], players[0, 2] = [0.0, 0.0], [6.0, 8.0]  # home: 5 m each from their centroid
    players[0, 1] = [1.0, 1.0]  # away: a lone player
    tracking = Tracking(np.zeros((1, 2), dtype=np.float32), players,
                        np.array([HOME, AWAY, HOME, AWAY], dtype=np.int8))
    f = features.compute(tracking, fps=25, pitch=PITCH)
    assert np.isclose(f.spread_home[0], 5.0) and np.isclose(f.spread_away[0], 0.0)


def test_ball_velocity_change_spikes_at_a_kick():
    ball = np.zeros((20, 2), dtype=np.float32)
    ball[:10, 0] = 0.0  # still until frame 9 ...
    ball[10:, 0] = np.arange(1, 11) * 0.8  # ... then 20 m/s along x
    tracking = Tracking(ball, np.zeros((20, 1, 2), dtype=np.float32),
                        np.array([HOME], dtype=np.int8))
    change = features.compute(tracking, fps=25, pitch=PITCH).ball_velocity_change
    assert np.nanargmax(change) in (9, 10) and np.nanmax(change) > 15.0
    assert np.isnan(change[:3]).all() and change[5] < 1.0
