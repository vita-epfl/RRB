import json
from .data import SceneRow, TrackRow


def trajnet_tracks(row):
    x = round(row.x, 2)
    y = round(row.y, 2)
    return json.dumps({'track': {'f': row.frame, 'p': row.pedestrian, 'x': x, 'y': y}})


def trajnet_scenes(row):
    return json.dumps(
        {'scene': {'id': row.scene, 'p': row.pedestrian, 's': row.start, 'e': row.end}})


def trajnet(row):
    if isinstance(row, TrackRow):
        return trajnet_tracks(row)
    elif isinstance(row, SceneRow):
        return trajnet_scenes(row)

    raise Exception('unknown row type')
