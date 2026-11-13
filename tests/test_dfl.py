import numpy as np
import pytest

from possession.dfl import DFL_DIR, load_dfl_match
from possession.official import A, D, H, X


def test_load_dfl_match_reads_its_cache_without_kloppy(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    truth = np.array([H, H, D, X, A, A], dtype=np.int8)
    np.savez(cache / "TEST.npz", ball=np.zeros((6, 2), np.float32),
             players=np.zeros((6, 2, 2), np.float32), team=np.array([0, 1], np.int8),
             truth=truth, runs=np.array([(0, 3), (4, 6)]), pitch=np.array([105.0, 68.0]))
    tracking, got, runs, pitch = load_dfl_match("TEST", root=tmp_path)
    assert runs == ((0, 3), (4, 6)) and pitch == (105.0, 68.0)
    assert np.array_equal(got, truth) and tracking.players.shape == (6, 2, 2)


@pytest.mark.skipif(not list(DFL_DIR.glob("*positions*J03WMX.xml")),
                    reason="DFL open data not downloaded (external/dfl)")
def test_a_real_dfl_match_has_two_halves_of_labelled_frames():
    tracking, truth, runs, _ = load_dfl_match("J03WMX")
    assert len(runs) == 2 and truth[runs[0][1]] == X
    labelled = truth != X
    assert labelled.sum() > 25 * 60 * 85  # over 85 minutes
    assert set(np.unique(truth[labelled])) == {H, A, D}
    assert tracking.players.shape[0] == len(truth)
