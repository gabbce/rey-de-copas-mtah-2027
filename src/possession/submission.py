"""Write the frame,label CSV and check it with the official validator."""

from pathlib import Path

import numpy as np

from possession.official import N_FRAMES, pm


def write_submission(path, codes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.array(pm.LABELS)[codes]
    # Explicit newline and encoding: the rules want the same file, byte for byte, on any OS.
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def validate(path, n_frames=N_FRAMES):
    """None if the official scorer would accept the file, else its one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_frames)
