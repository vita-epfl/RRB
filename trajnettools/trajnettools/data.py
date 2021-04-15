from collections import namedtuple


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y'])
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end'])
