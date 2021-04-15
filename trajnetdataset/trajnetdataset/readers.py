import json
import os
import xml.etree.ElementTree

import numpy as np
import scipy.interpolate
import pdb
from trajnettools import TrackRow


def biwi(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0]) - 1),  # shift from 1-index to 0-index
                    int(float(line[1])),
                    float(line[2]),
                    float(line[4]))

def honda(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[1])),
                    float(line[2]),
                    float(line[3]))
                    
def CarRacing(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[1])),
                    float(line[3]),
                    float(line[2]))
                    
def epflRoundabout(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[2])),
                    float(line[4]),  # because x, y are different in their code.
                    float(line[3]))
                    
def interaction(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[1])),
                    float(line[3]),
                    float(line[2]))                    

def nuScene(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[1])),
                    float(line[2]),  # because x, y are different in their code.
                    float(line[3]))
					
def synthetic(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])), 
                    int(float(line[1])),
                    float(line[2]),  # because x, y are different in their code.
                    float(line[3]))
def crowds_interpolate_person(ped_id, person_xyf):
    xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12  # TODO
    ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12
    fs = np.array([f for _, _, f in person_xyf])

    kind = 'linear'
    if len(fs) > 5:
        kind = 'cubic'

    x_fn = scipy.interpolate.interp1d(fs, xs, kind=kind)
    y_fn = scipy.interpolate.interp1d(fs, ys, kind=kind)

    frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)
    return [TrackRow(int(f), ped_id, x, y)
            for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T]


def crowds(whole_file):
    pedestrians = []
    current_pedestrian = []
    for line in whole_file.split('\n'):
        if '- Num of control points' in line or \
           '- the number of splines' in line:
            if current_pedestrian:
                pedestrians.append(current_pedestrian)
            current_pedestrian = []
            continue

        # strip comments
        if ' - ' in line:
            line = line[:line.find(' - ')]

        # tokenize
        entries = [e for e in line.split(' ') if e]
        if len(entries) != 4:
            continue

        x, y, f, _ = entries
        current_pedestrian.append([float(x), float(y), int(f)])

    if current_pedestrian:
        pedestrians.append(current_pedestrian)

    return [row
            for i, p in enumerate(pedestrians)
            for row in crowds_interpolate_person(i, p)]


def mot_xml(file_name):
    """PETS2009 dataset.

    Original frame rate is 7 frames / sec.
    """
    tree = xml.etree.ElementTree.parse(file_name)
    root = tree.getroot()
    for frame in root:
        f = int(frame.attrib['number'])
        if f % 2 != 0:  # reduce to 3.5 rows / sec
            continue

        for ped in frame.find('objectlist'):
            p = ped.attrib['id']
            box = ped.find('box')
            x = box.attrib['xc']
            y = box.attrib['yc']

            yield TrackRow(f, int(p), float(x) / 100.0, float(y) / 100.0)


def mot(line):
    """Line reader for MOT files.

    MOT format:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    line = [e for e in line.split(',') if e != '']
    return TrackRow(int(float(line[0])),
                    int(float(line[1])),
                    float(line[7]),
                    float(line[8]))


def edinburgh(filename_content_index):
    """Edinburgh Informatics Forum data reader.

    Original frame rate is 9fps.
    Every pixel corresponds to 24.7mm.
    http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/
    """
    (_, whole_file), index = filename_content_index

    for line in whole_file.splitlines():
        line = line.strip()
        if not line.startswith('TRACK.R'):
            continue

        # get to track id
        line = line[7:]
        track_id, _, coordinates = line.partition('=')
        track_id = int(track_id) + index * 1000000

        # parse track
        for coordinates in coordinates.split(';'):
            if not coordinates:
                continue
            x, y, frame = coordinates.strip('[] ').split(' ')
            frame = int(frame) + index * 1000000
            if frame % 3 != 0:  # downsample frame rate
                continue
            yield TrackRow(frame, track_id, float(x) * 0.0247, float(y) * 0.0247)


def syi(filename_content):
    """Tracking dataset in Grand Central.

    Yi_Pedestrian_Travel_Time_ICCV_2015_paper.pdf states that original
    frame rate is 25fps.

    Input rows are sampled every 20 frames. Assuming 25fps at recording,
    need to interpolate an additional row to get to 2.5 rows per second.
    """
    filename, whole_file = filename_content
    track_id = int(os.path.basename(filename).replace('.txt', ''))

    chunk = []
    last_row = None
    for line in whole_file.split('\n'):
        if not line:
            continue
        chunk.append(int(line))
        if len(chunk) < 3:
            continue

        # rough approximation of mapping to world coordinates (main concourse is 37m x 84m)
        new_row = TrackRow(chunk[2], track_id, chunk[0] * 30.0 / 1920, chunk[1] * 70.0 / 1080)

        # interpolate one row to increase frame rate
        if last_row is not None:
            interpolated_row = TrackRow(
                int((last_row.frame + new_row.frame) / 2),
                track_id,
                (last_row.x + new_row.x) / 2,
                (last_row.y + new_row.y) / 2,
            )
            yield interpolated_row

        yield new_row
        chunk = []
        last_row = new_row


def dukemtmc(input_array, query_camera=5):
    """DukeMTMC dataset.

    Recorded at 59.940059 fps.

    Line format:
    [camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY]
    """
    for line in input_array:
        camera, person, frame, _, _, _, _, world_x, world_y, _, _ = line

        camera = int(camera)
        if camera != query_camera:
            continue

        frame = int(frame)
        if frame % 24 != 0:
            continue

        yield TrackRow(frame, int(person), world_x, world_y)


def wildtrack(filename_content):
    filename, content = filename_content

    frame = int(os.path.basename(filename).replace('.json', ''))
    for entry in json.loads(content):
        ped_id = entry['personID']
        position_id = entry['positionID']

        x = -3.0 + 0.025 * (position_id % 480)
        y = -9.0 + 0.025 * (position_id / 480)

        yield TrackRow(frame, ped_id, x, y)


def trajnet_original(line):
    line = [e for e in line.split(' ') if e != '']
    return TrackRow(int(float(line[0])),
                    int(float(line[1])),
                    float(line[2]),
                    float(line[3]))
