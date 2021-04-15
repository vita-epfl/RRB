from collections import defaultdict

import trajnettools
from trajnettools import SceneRow
import pdb
import math

class Scenes(object):
    def __init__(self, start_scene_id=0, chunk_size=21, chunk_stride=5, visible_chunk=None):
        self.scene_id = start_scene_id
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.visible_chunk = visible_chunk
        self.frames = set()
        self.obs = 9
    @staticmethod
    def euclidean_distance(row1, row2):
        """Euclidean distance squared between two rows."""
        #pdb.set_trace()
        return math.sqrt((row1.x - row2.x)**2 + (row1.y - row2.y)**2)	# Modified by Hoseein: Why we don't have sqrt? it is in power 2!

    @staticmethod
    def close_pedestrians(rows, cell_size=1000): # svene's number was 3
        """Fast computation of spatially close pedestrians.

        By frame, get the list of pedestrian ids that or close to other
        pedestrians. Approximate with multi-occupancy of discrete grid cells.
        """
        sparse_occupancy = defaultdict(list)
        for row in rows:
            x = int(row.x // cell_size * cell_size)
            y = int(row.y // cell_size * cell_size)
            sparse_occupancy[(x, y)].append(row.pedestrian)
        return {ped_id
                for cell in sparse_occupancy.values() if len(cell) > 1
                for ped_id in cell}

    @staticmethod
    def continuous_frames(frames, tolerance=1.5):
        increments = [f2 - f1 for f1, f2 in zip(frames[:-1], frames[1:])]
        median_increment = sorted(increments)[int(len(increments) / 2)]
        ok = median_increment * tolerance > max(increments)

        if not ok:
            print('!!!!!!!!! DETECTED GAP IN FRAMES')
            print(increments)

        return ok

    def from_rows(self, rows):
        count_by_frame = rows.groupBy(lambda r: r.frame).mapValues(len).collectAsMap()
        occupancy_by_frame = (rows
                              .groupBy(lambda r: r.frame)
                              .mapValues(self.close_pedestrians)
                              .collectAsMap())

        def to_scene_row(ped_frames):
            ped_id, scene_frames = ped_frames
            row = SceneRow(self.scene_id, ped_id, scene_frames[0], scene_frames[-1])
            self.scene_id += 1
            return row

        scenes = (
            rows
            .groupBy(lambda r: r.pedestrian)
            .filter(lambda p_path: len(p_path[1]) >= self.chunk_size)
            .mapValues(lambda path: sorted(path, key=lambda p: p.frame))
            .flatMapValues(lambda path: [
                [path[ii].frame for ii in range(i, i + self.chunk_size)]
                for i in range(0, len(path) - self.chunk_size - 1, self.chunk_stride)
                # filter for pedestrians moving by more than 50 meter
                #if self.euclidean_distance_2(path[i], path[i+self.chunk_size-1]) >  self.chunk_size/4 #self.chunk_size/2    #avg speed is around 4 per frame so self.chunk_size/2 is a small number to ensure there is a movement
                #if ((self.euclidean_distance(path[i], path[i+self.chunk_size-1]) >  self.chunk_size*2) #self.chunk_size/4    #avg speed is around 4 per frame so self.chunk_size/2 is a small number to ensure there is a movement
                #and (self.euclidean_distance(path[i], path[i+self.obs-1]) >  self.obs*2)) #keep the onses that we have movements in the observation part
            ])

            # filter out scenes with large gaps in frame numbers
            .filter(lambda ped_frames: self.continuous_frames(ped_frames[1]))

            # filter for scenes that have some activity
            .filter(lambda ped_frames:
                    sum(count_by_frame[f] for f in ped_frames[1]) >= 1.0 * self.chunk_size) #2.0 * self.chunk_size)

            # require some proximity to other pedestrians
            #.filter(lambda ped_frames:
            #        ped_frames[0] in {p
            #                          for frame in ped_frames[1]
            #                          for p in occupancy_by_frame[frame]})

            .cache()
        )

        self.frames |= set(scenes
                           .flatMap(lambda ped_frames:
                                    ped_frames[1]
                                    if self.visible_chunk is None
                                    else ped_frames[1][:self.visible_chunk])
                           .toLocalIterator())
        return scenes.map(to_scene_row)


    def rows_to_file(self, rows, output_file):
        #pdb.set_trace()
        scenes = self.from_rows(rows)
        tracks = rows.filter(lambda r: r.frame in self.frames)
        all_data = rows.context.union((scenes, tracks))
        all_data.map(trajnettools.writers.trajnet).saveAsTextFile(output_file)

        return self
