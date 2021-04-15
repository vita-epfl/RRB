from collections import defaultdict
import itertools
import json
import random
import pdb
import numpy as np

from .data import SceneRow, TrackRow

import pdb

class Reader(object):
    """Read trajnet files.

    :param scene_type: None -> numpy.array, 'rows' -> TrackRow and SceneRow, 'paths': grouped rows (primary pedestrian first)
    """
    def __init__(self, input_file, scene_type=None):
        if scene_type is not None and scene_type not in {'rows', 'paths'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()

        self.read_file(input_file)
        self.input_file = input_file
        file = open('./segmentedImgs/frame_rate.txt')         # generate dictionary for frame rate
        lines = file.read().split()
        self.frame_rate_dict = dict(zip(lines[::2], lines[1::2]))   
        
    def read_file(self, input_file):
        #pdb.set_trace()
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:                   				    
                    row = TrackRow(track['f'], track['p'], track['x'], track['y'])
                    #if input_file=='../trajnetdataset/output/train/HondaFunabori.ndjson' and row.frame==2502:	    
                        #pdb.set_trace()
                    self.tracks_by_frame[row.frame].append(row)
                    #pdb.set_trace()
                    continue
                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'])
                    self.scenes_by_id[row.scene] = row
                    
    def scenes(self, randomize=False, limit=0, ids=None, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if ids is not None:
            scene_ids = ids
        if randomize:
            scene_ids = list(scene_ids)
            random.shuffle(scene_ids)
        if limit:
            scene_ids = itertools.islice(scene_ids, limit)
        if sample is not None:
            scene_ids = random.sample(scene_ids, int(len(scene_ids) * sample))
        for scene_id in scene_ids:
            yield self.scene(scene_id)

    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        paths = defaultdict(list)
        for row in track_rows:
            paths[row.pedestrian].append(row)
        #if(track_rows[1].frame == 885):
         #   pdb.set_trace()
        # list of paths with the first path being the path of the primary pedestrian
        primary_path = paths[primary_pedestrian]
        other_paths = [path for ped_id, path in paths.items() if ped_id != primary_pedestrian]
        return [primary_path] + other_paths

    @staticmethod
    def paths_to_xy(paths):
        frames = [r.frame for r in paths[0]] # paths[0] is path of ped of interest. frames will be the list of frames in the scene
        pedestrians = [path[0].pedestrian for path in paths] #pedestrians will be the list of pedestrians in the scene

        frame_to_index = {frame: i for i, frame in enumerate(frames)}
        xy = np.full((len(frames), len(pedestrians), 2), np.nan)

        for ped_index, path in enumerate(paths):
            for indx , row in enumerate(path):
                #if(indx==16 and path[indx].pedestrian==20327):
                 #   pdb.set_trace()                    
                if row.frame not in frame_to_index:
                    continue
                entry = xy[frame_to_index[row.frame]][ped_index]
                entry[0] = row.x
                entry[1] = row.y
        #pdb.set_trace()
        return xy

    def scene(self, scene_id):
        scene = self.scenes_by_id.get(scene_id)
        #remove address of the file and preserve the file name
        for i in range(len(self.input_file)):
            if(self.input_file[-i-1]=='/'):
                break
        file_name = self.input_file[-i:-7]  
                
        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]
        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.pedestrian, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.pedestrian, track_rows) # returns the paths(frames from first to end of a specific pedestrian) of different pedestrians, the first one is the path of ped on interest and then other ones.(so it is [[trajnettools.data.Rows of ped interest],[trajnettools.data.Row of next ped], ...]]    
        if(paths[0][1].frame-paths[0][0].frame == 0):
            pdb.set_trace()
        sample_rate = int(self.frame_rate_dict[file_name])/(paths[0][1].frame-paths[0][0].frame) # frame per second devided by annotation per frame
        if self.scene_type == 'paths':
            return file_name, scene_id, paths, sample_rate
                
        # return a numpy array
        return file_name, scene_id, self.paths_to_xy(paths), sample_rate
