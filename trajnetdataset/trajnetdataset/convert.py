"""Create Trajnet data from original datasets."""

import subprocess
import argparse
import pysparkling
import scipy.io
import trajnettools
from . import readers
from .scene import Scenes
import os
import pdb

def biwi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.biwi)
            .cache())

def honda(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.honda)
            .cache())
def epflRoundabout(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.epflRoundabout)
            .cache())
def nuScene(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.nuScene)
            .cache())
def crowds(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .values()
            .flatMap(readers.crowds)
            .cache())
def synthetic(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.synthetic)
            .cache())

def CarRacing(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.CarRacing)
            .cache())

def interaction(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.interaction)
            .cache())            
            
def write_train(input_rows, output_file, both_train_and_val_data, train_fraction=0.70, chunk_size=21, train_chunk_stride=11, val_chunk_stride=1, visible_chunk=9, frame_resampling_rate = 1):
    frames = sorted(set(input_rows.filter(lambda r: r.frame%frame_resampling_rate==0).map(lambda r: r.frame).toLocalIterator())) # resampling is added to change distance between frames
    train_split_index = int(len(frames) * train_fraction)
    train_frames = set(frames[:train_split_index])
    val_monitor_frames = set(frames[train_split_index:])
    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    if os.path.isfile(train_output): #if file exists, remove it to prevent error
        subprocess.call(["rm",train_output]) 
    #pdb.set_trace()
    train_scenes = Scenes(chunk_size=chunk_size, chunk_stride=train_chunk_stride).rows_to_file(train_rows, train_output)

def write_val(input_rows, output_file, chunk_size=21, visible_chunk=9, frame_resampling_rate = 1, val_chunk_stride=1):
    frames = sorted(set(input_rows.filter(lambda r: r.frame % frame_resampling_rate == 0).map(
        lambda r: r.frame).toLocalIterator()))  # resampling is added to change distance between frames
    # val_frames = set(frames)
    '''val_frames = set(frames[:int(len(frames) * 0.05)])
    val_monitor_split_index = int(len(frames) * 0.01)
    val_monitor_frames = set(frames[:val_monitor_split_index])'''
    
    train_split_index = int(len(frames) * 0.7)
    val_monitor_split_index = int(len(frames) * 0.1)
    val_monitor_frames = set(frames[train_split_index:train_split_index+val_monitor_split_index])
    val_frames = set(frames[train_split_index+val_monitor_split_index:])

    # validation set:
    test_rows = input_rows.filter(lambda r: r.frame in val_frames)
    test_rows_val = input_rows.filter(lambda r: r.frame in val_monitor_frames)
    test_output = output_file.format(split='val')
    test_for_val_output = output_file.format(split='val_for_monitoring_training')
    # visible_chunk measures number of frames for observation
    if os.path.isfile(test_output):  # if file exists, remove it to prevent error
        subprocess.call(["rm", test_output])
    if os.path.isfile(test_for_val_output):  # if file exists, remove it to prevent error
        subprocess.call(["rm", test_for_val_output])
    test_scenes = Scenes(chunk_size=chunk_size, chunk_stride=val_chunk_stride).rows_to_file(test_rows, test_output)
    test_scenes = Scenes(chunk_size=chunk_size, chunk_stride=val_chunk_stride).rows_to_file(test_rows_val,
                                                                                            test_for_val_output)

def main():
    sc = pysparkling.Context()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obs', default=9, type=int,
                        help='observation length in frames')
    parser.add_argument('--n_pred', default=12, type=int,
                        help='prediction length in frames')   
    parser.add_argument('--train_chunk_stride', default=10, type=int,
                        help='the distance between two chuncks which is considered for observation and prediction in train data')  
    parser.add_argument('--val_chunk_stride', default=1, type=int,
                        help='the distance between two chuncks which is considered for observation and prediction in validation data')
    parser.add_argument('--val_monitor_fraction', default=0.3, type=float,
                        help='fraction of the data to be used for monitoring training')

    args = parser.parse_args()
    #First remove all previous files to avoid mixing and then generate active files
    for baseAdd in ['../trajnetdataset/output/train/', '../trajnetdataset/output/val_for_monitoring_training/', '../trajnetdataset/output/val/']:
        list = os.listdir(baseAdd) 
        prevFiles = [i for i in list]
        for i in prevFiles:
            subprocess.call(["rm",baseAdd+i])
    
    scenes = []
    train_scenes = []
    val_scenes = []
    files = []
    train_files = []
    val_files = []
    
    train_data = {#'Ecublens_Route_de_la_Pierre_1_tracked_all.txt', 'Ecublens_Route_de_la_Pierre_2_tracked_all.txt'
                  #,'Echabdens_Route_de_la_Gare_1_tracked_all.txt', 'Echabdens_Route_de_la_Gare_2_tracked_all.txt', 'Echabdens_Route_de_la_Gare_3_tracked_all.txt', 'Echabdens_Route_de_la_Gare_4_tracked_all.txt', 'Echabdens_Route_de_la_Gare_5_tracked_all.txt'
                  #,'EPFL_Route_Cantonale_31_1_tracked_all.txt','EPFL_Route_Cantonale_31_2_tracked_all.txt'
                  #,'Morges_Avenue_de_la_Gotta_1_tracked_all.txt','Morges_Avenue_de_la_Gotta_2_tracked_all.txt','Morges_Avenue_de_la_Gotta_3_tracked_all.txt','Morges_Avenue_de_la_Gotta_5_tracked_all.txt'
                  'DJI_0001-Roseville-annot1Point.txt','DJI_0002-Roseville-annot1Point.txt', 'DJI_0003-Roseville-annot1Point.txt', 'DJI_0004-Roseville-annot1Point.txt'
                  #,'DJI_0001-UC_Davis-annot1Point.txt', 'DJI_0002-UC_Davis-annot1Point.txt', 'DJI_0003-UC_Davis-annot1Point.txt', 'DJI_0004-UC_Davis-annot1Point.txt'
                  ,'DJI_0001-Petaluma-annot1Point.txt', 'DJI_0002-Petaluma-annot1Point.txt', 'DJI_0003-Petaluma-annot1Point.txt', 'DJI_0004-Petaluma-annot1Point.txt'
                  ,'DJI_0001-GrassValleySierraCollege-annot1Point.txt','DJI_0002-GrassValleySierraCollege-annot1Point.txt','DJI_0003-GrassValleySierraCollege-annot1Point.txt','DJI_0004-GrassValleySierraCollege-annot1Point.txt','DJI_0005-GrassValleySierraCollege-annot1Point.txt'
                  #'road_009.txt','road_001.txt','road_002.txt','road_003.txt','road_004.txt','road_005.txt'
                   #'DJI_0001-Petaluma-annot1Point.txt','DJI_0002-GrassValleySierraCollege-annot1Point.txt'
                   #'DJI_0001-Roseville-annot1Point.txt','DJI_0001-Petaluma-annot1Point.txt','DJI_0002-GrassValleySierraCollege-annot1Point.txt'}
                }
    both_train_and_val_data = {} #DJI_0003-Roseville-annot1Point.txt
    val_data = {}
    
    #interaction data   
    baseAdd = '../trajnetdataset/data/raw/interaction/active-files-train/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    resampling = 5 # every 5 frames, sample 1 -> it will be 2 frame/sec, similar to Social-WaGDAT paper
    for file in valFiles:    
        train_scenes.append(interaction(sc, baseAdd+file))
        train_files.append('resampling'+str(resampling)+'-'+file)
        
    #baseAdd = '../trajnetdataset/data/raw/interaction/active-files-val/' 
    baseAdd = '../trajnetdataset/data/raw/interaction/active-files-train/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list]
    resampling = 5 # every 5 frames, sample 1 
    for file in valFiles:
        val_scenes.append(interaction(sc, baseAdd+file))
        val_files.append('resampling'+str(resampling)+'-'+file)
        
    '''#Honda roundabout
    baseAdd = '../trajnetdataset/data/raw/Honda/honda_roundabout_annotations/'  
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list]
    resampling = 4 # every 1 frames, sample 1 
    for file in valFiles:
        if(file in train_data):
            train_scenes.append(honda(sc, baseAdd+file))
            train_files.append('resampling'+str(resampling)+'-'+file)
        if(file in val_data):
            val_scenes.append(honda(sc, baseAdd+file))
            val_files.append('resampling'+str(resampling)+'-'+file)
    
    #synthetic data   
    baseAdd = '../trajnetdataset/data/raw/synthetic/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list]
    resampling = 1 # every 1 frames, sample 1 
    for file in valFiles:
        #scenes.append(synthetic(sc, baseAdd+file))
        #files.append('resampling'+str(resampling)+'-'+file)        
        if(file in train_data):
            train_scenes.append(synthetic(sc, baseAdd+file))
            train_files.append('resampling'+str(resampling)+'-'+file)
        if(file in val_data):
            val_scenes.append(synthetic(sc, baseAdd+file))
            val_files.append('resampling'+str(resampling)+'-'+file)
  
    #CarRacing data   
    
    baseAdd = '../trajnetdataset/data/raw/CarRacing_multimodal/active-files-train/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    resampling = 1 # every 3 frames, sample 1 -> it will be 5 frame/sec
    for file in valFiles:    
        train_scenes.append(CarRacing(sc, baseAdd+file))
        train_files.append('resampling'+str(resampling)+'-'+file)
        
    baseAdd = '../trajnetdataset/data/raw/CarRacing_multimodal/active-files-val/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list]
    resampling = 1 # every 1 frames, sample 1 
    for file in valFiles:
        val_scenes.append(CarRacing(sc, baseAdd+file))
        val_files.append('resampling'+str(resampling)+'-'+file)
    
            
    #EPFLRoundabout data   
    baseAdd = '../trajnetdataset/data/raw/EPFLRoundabout/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list]
    resampling = 2 # every 3 frames, sample 1 -> it will be 5 frame/sec
    for file in valFiles:
        #scenes.append(synthetic(sc, baseAdd+file))
        #files.append('resampling'+str(resampling)+'-'+file)        
        if(file in train_data):
            train_scenes.append(epflRoundabout(sc, baseAdd+file))
            train_files.append('resampling'+str(resampling)+'-'+file)
        if(file in val_data):
            val_scenes.append(epflRoundabout(sc, baseAdd+file))
            val_files.append('resampling'+str(resampling)+'-'+file)'''
   
    ''' #EPFLRoundabout
    baseAdd = '../trajnetdataset/data/raw/EPFLRoundabout/active-files/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    resampling = 3 # every 3 frames, sample 1 -> it will be 5 frame/sec
    for file in valFiles:    
        scenes.append(epflRoundabout(sc, baseAdd+file))
        files.append('resampling'+str(resampling)+'-'+file)
    
    baseAdd = '../trajnetdataset/data/raw/nuScene/active-files/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    for file in valFiles:    
        scenes.append(nuScene(sc, baseAdd+file))
        files.append(file)    


    baseAdd = '../trajnetdataset/data/raw/biwi/active-files/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    for file in valFiles:    
        scenes.append(biwi(sc, baseAdd+file))
        files.append(file)
    
    baseAdd = '../trajnetdataset/data/raw/crowds/active-files/' 
    print('Reading raw files in', baseAdd, '...')
    list = os.listdir(baseAdd) 
    valFiles = [i for i in list] 
    for file in valFiles:    
        scenes.append(crowds(sc, baseAdd+file))
        files.append(file)'''
        
    for i,scene in enumerate(train_scenes):
        frame_resampling_rate = 1
        if 'resampling' in train_files[i][0:-4]:
            frame_resampling_rate = int(train_files[i][10])
            train_files[i] = train_files[i][12:]
        write_train(scene,'output/{split}/'+train_files[i][0:-4]+'.ndjson',chunk_size=args.n_obs+args.n_pred, train_chunk_stride=args.train_chunk_stride,\
        both_train_and_val_data=both_train_and_val_data, val_chunk_stride=args.val_chunk_stride, visible_chunk=args.n_obs, frame_resampling_rate=frame_resampling_rate)
    for i,scene in enumerate(val_scenes):
        frame_resampling_rate = 1
        if 'resampling' in val_files[i][0:-4]:
            frame_resampling_rate = int(val_files[i][10])
            val_files[i] = val_files[i][12:]
        write_val(scene,'output/{split}/'+val_files[i][0:-4]+'.ndjson',chunk_size=args.n_obs+args.n_pred, val_chunk_stride=args.val_chunk_stride, visible_chunk=args.n_obs,\
        frame_resampling_rate=frame_resampling_rate)
    
if __name__ == '__main__':
    main()
