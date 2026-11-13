import json

import numpy as np

from possession.official import A, D, H, X
from possession.sample_game import PITCH, possession_truth, read_sample_tracking
from possession.tracking import AWAY, HOME

# Two stretches of frames with the two tracked players in opposite group orders, as substitutions
# make the real file do.
META = """<main><Metadata>
<Teams><Team id="TA"><Name>Team A</Name></Team><Team id="TB"><Name>Team B</Name></Team></Teams>
<Players><Player id="P1" teamId="TA"/><Player id="P2" teamId="TB"/></Players>
<PlayerChannels>
<PlayerChannel channelId="x" id="player1_x" playerId="P1"/>
<PlayerChannel channelId="y" id="player1_y" playerId="P1"/>
<PlayerChannel channelId="x" id="player2_x" playerId="P2"/>
<PlayerChannel channelId="y" id="player2_y" playerId="P2"/>
</PlayerChannels>
</Metadata><DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="2"><SplitRegister>
<SplitRegister><PlayerChannelRef playerChannelId="player1_x"/>
<PlayerChannelRef playerChannelId="player1_y"/></SplitRegister>
<SplitRegister><PlayerChannelRef playerChannelId="player2_x"/>
<PlayerChannelRef playerChannelId="player2_y"/></SplitRegister></SplitRegister>
</DataFormatSpecification>
<DataFormatSpecification startFrame="3" endFrame="4"><SplitRegister>
<SplitRegister><PlayerChannelRef playerChannelId="player2_x"/>
<PlayerChannelRef playerChannelId="player2_y"/></SplitRegister>
<SplitRegister><PlayerChannelRef playerChannelId="player1_x"/>
<PlayerChannelRef playerChannelId="player1_y"/></SplitRegister></SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications></main>"""

RAW = ("1:0.1,0.1;0.9,0.9:0.5,0.5\n2:0.1,0.1;0.9,0.9:NaN,NaN\n"
       "3:0.9,0.9;0.1,0.1:0.5,0.5\n4:0.9,0.9;0.1,0.1:0.5,0.5\n")


def test_sample_tracking_follows_each_stretch_s_group_order(tmp_path):
    meta, raw = tmp_path / "meta.xml", tmp_path / "tracking.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    tracking, n_frames = read_sample_tracking(meta, raw)
    assert n_frames == 4 and tracking.team.tolist() == [HOME, AWAY]
    # P1 stays at (0.1, 0.1) all along, although its group moves from first to second.
    assert np.allclose(tracking.players[:, 0], [[0.1 * PITCH[0], 0.1 * PITCH[1]]] * 4)
    assert np.isnan(tracking.ball[1]).all() and np.allclose(tracking.ball[2], [52.5, 34.0])


def event(index, kind, frame, team, subtypes=None, end=None, period=1):
    return {"index": index, "type": {"name": kind}, "team": {"name": team},
            "subtypes": subtypes, "period": period,
            "start": {"frame": frame}, "end": {"frame": end or frame}}


def test_possession_truth_follows_control_events_and_stoppages(tmp_path):
    events = [
        event(1, "SET PIECE", 11, "Team A", {"name": "KICK OFF"}),
        event(2, "PASS", 15, "Team A", end=20),
        event(3, "BALL LOST", 30, "Team A"),
        event(4, "RECOVERY", 31, "Team B"),
        event(5, "BALL OUT", 40, "Team B", end=45),
        event(6, "SET PIECE", 61, "Team A", {"name": "THROW IN"}),
        event(7, "CARRY", 62, "Team A", end=70),
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"data": events}))
    codes, runs = possession_truth(path, 80)
    assert runs == ((10, 70),)
    assert (codes[:10] == X).all() and (codes[70:] == X).all()
    assert (codes[10:30] == H).all()  # a lost ball stays with its team until the rival controls
    assert (codes[30:44] == A).all()
    assert (codes[44:60] == D).all()  # from the ball going out to the throw-in
    assert (codes[60:70] == H).all()


HEADER = ",,,{side},,{side},,,\n,,,1,,2,,,\nPeriod,Frame,Time [s],Player1,,Player2,,Ball,\n"


def test_csv_tracking_joins_both_teams_and_takes_the_ball_once(tmp_path):
    from possession.sample_game import read_csv_tracking
    home = tmp_path / "home.csv"
    away = tmp_path / "away.csv"
    home.write_text(HEADER.format(side="Home")
                    + "1,1,0.04,0.1,0.1,NaN,NaN,0.5,0.5\n1,2,0.08,0.2,0.2,0.3,0.3,NaN,NaN\n")
    away.write_text(HEADER.format(side="Away")
                    + "1,1,0.04,0.9,0.9,0.8,0.8,0.5,0.5\n1,2,0.08,0.9,0.9,0.8,0.8,NaN,NaN\n")
    tracking, n_frames = read_csv_tracking(home, away)
    assert n_frames == 2 and tracking.team.tolist() == [HOME, HOME, AWAY, AWAY]
    assert np.allclose(tracking.players[0, 0], [0.1 * PITCH[0], 0.1 * PITCH[1]])
    assert np.isnan(tracking.players[0, 1]).all() and np.isnan(tracking.ball[1]).all()


def test_csv_events_derive_the_same_truth_as_json(tmp_path):
    from possession.sample_game import read_csv_events
    path = tmp_path / "events.csv"
    path.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
                    "From,To,Start X,Start Y,End X,End Y\n"
                    "Home,SET PIECE,KICK OFF,1,11,0.4,11,0.4,P1,,,,,\n"
                    "Home,PASS,,1,15,0.6,20,0.8,P1,P2,,,,\n"
                    "Away,RECOVERY,INTERCEPTION,1,31,1.2,31,1.2,P9,,,,,\n"
                    "Away,BALL OUT,,1,40,1.6,45,1.8,P9,,,,,\n"
                    "Home,PASS,GOAL KICK,1,61,2.4,70,2.8,P1,P3,,,,\n")
    events = read_csv_events(path)
    assert events[2].subtypes == {"INTERCEPTION"}
    codes, runs = possession_truth(path, 80)
    assert runs == ((10, 70),)
    assert (codes[10:30] == H).all() and (codes[30:44] == A).all()
    assert (codes[44:60] == D).all() and (codes[60:70] == H).all()
