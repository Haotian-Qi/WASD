import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile 
import sys
import argparse
import pandas as pd
import pickle 

import torch

def extract_audio(orig_vid_dir, orig_audio_dir):
    # Take 1 hour to extract the audio from movies
    # for dataType in ['trainval', 'test']:
    for dataType in ['trainval']:
        inpFolder = '%s/%s'%(orig_vid_dir, dataType)
        outFolder = '%s/%s'%(orig_audio_dir, dataType)
        os.makedirs(outFolder, exist_ok = True)
        videos = glob.glob("%s/*"%(inpFolder))
        for videoPath in tqdm.tqdm(videos):

            if videoPath.split('/')[-1].split('.')[0] == 'AFNYkYz9Cqw_100-130':
                audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
                cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
                subprocess.call(cmd, shell=True, stdout=None)

def extract_audio_clips(dataset_sets, csv_dir, clip_audio_dir, orig_audio_dir):

    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}

    for dataType in dataset_sets:
        df = pandas.read_csv(os.path.join(csv_dir, '%s_orig.csv'%(dataType)), engine='python')
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}
        outDir = os.path.join(clip_audio_dir, dataType)
        audioDir = os.path.join(orig_audio_dir, dic[dataType])
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)

        for entity in tqdm.tqdm(entityList, total=len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            if videoKey == 'AFNYkYz9Cqw_100-130':
                videoName = insData.iloc[0]['video_id'][:11]  # Assuming this extracts the videoName correctly
                start = insData.iloc[0]['frame_timestamp']
                end = insData.iloc[-1]['frame_timestamp']
                entityID = insData.iloc[0]['entity_id']
                # Change here: Use videoName for the folder instead of videoKey
                insPath = os.path.join(outDir, videoName, entityID + '.wav')

                if videoKey not in audioFeatures.keys():                
                    audioFile = os.path.join(audioDir, videoKey + '.wav')
                    sr, audio = wavfile.read(audioFile)
                    audioFeatures[videoKey] = audio
                    
                audioStart = int(float(start) * sr)
                audioEnd = int(float(end) * sr)
                audioData = audioFeatures[videoKey][audioStart:audioEnd]

                # # Ensure the directory exists before writing the file
                os.makedirs(os.path.dirname(insPath), exist_ok=True)
                wavfile.write(insPath, sr, audioData)

def write_annotation(dataset_sets, csv_dir, orig_vid_dir):

    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    header = ['videoName', 'entityID', 'fps', 'numFrames', 'facePos', 'labels']
    for dataType in dataset_sets:
        print("Extracting {} data of {} set...".format('face', dataType))
        csv_file = "{}_orig.csv".format(dataType)

        df = pandas.read_csv(os.path.join(csv_dir, csv_file))
        df = df.sort_values(['frame_timestamp', 'entity_id']).reset_index(drop=True)

        datalist = []

        for video_id, video_df in tqdm.tqdm(df.groupby('video_id'), desc='Processing Videos'):
            # Ensure directory exists
            labels = []
            pos = []
            entityID = []
            frame_index = []
            count = 0
            audio_index = []
            for frame_timestamp, frame_df in video_df.groupby('frame_timestamp'):
                audio_index.append(frame_timestamp)
                for _, row in frame_df.iterrows():
                    labels.append(row['label_id'])
                    x1 = row['entity_box_x1']
                    y1 = row['entity_box_y1']
                    x2 = row['entity_box_x2']
                    y2 = row['entity_box_y2']
                    pos.append([(x1 + x2) / 2, 
                              (y1 + y2) / 2,
                                numpy.sqrt((x2 - x1)**2 + (y2 - y1)**2)])
                    frame_index.append(round(frame_timestamp, 4))

                    count+=1
                    entityID.append(int(row['entity_id'][-1])-1)

            videoDir = os.path.join(orig_vid_dir, dic[dataType])
            videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(video_id)))[0]

            datalist.append([video_id, audio_index, entityID, frame_index, pos, labels])

        with open(f'WASD/csv/ID_{dataType}.pkl', 'wb') as file:
            pickle.dump(datalist, file)

def extract_video_clips(dataset_sets, csv_dir, clip_vid_dir, orig_vid_dir, extract_body_data=False):

    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    header = ['videoName', 'entityID', 'fps', 'numFrames', 'facePos', 'labels']
    for dataType in dataset_sets:
        print("Extracting {} data of {} set...".format('face', dataType))
        csv_file = "{}_orig.csv".format(dataType)

        df = pandas.read_csv(os.path.join(csv_dir, csv_file))
        df = df.sort_values(['frame_timestamp', 'entity_id']).reset_index(drop=True)

        outDir = os.path.join(clip_vid_dir, dataType)
        datalist = []

        for video_id, video_df in tqdm.tqdm(df.groupby('video_id'), desc='Processing Videos'):
            if video_id == 'AFNYkYz9Cqw_100-130':
                video_dir = os.path.join(outDir, video_id)
                # Ensure directory exists
                num_frames = video_df['entity_id'].value_counts().max()
                os.makedirs(video_dir, exist_ok=True)

                videoDir = os.path.join(orig_vid_dir, dic[dataType])
                videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(video_id)))[0]

                insDir = os.path.join(os.path.join(outDir, video_id))
                if not os.path.isdir(insDir):
                    os.makedirs(insDir)

                labels = []
                pos = []
                entityID = []
                frame_index = []
                count = 0
                for frame_timestamp, frame_df in video_df.groupby('frame_timestamp'):

                    # entityID = frame_df.iloc[0]['entity_id']
                    j = 0
                    # videoName, entityID, fps, numFrames, position, labels
                    for _, row in frame_df.iterrows():
                        # entityID = row['entity_id']
                        labels.append(row['label_id'])
                        pos.append([row['entity_box_x1'], row['entity_box_y1'],
                                    row['entity_box_x2'], row['entity_box_y2']])
                        frame_index.append(round(frame_timestamp, 2))
                        videoDir = os.path.join(orig_vid_dir, dic[dataType])
                        videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(video_id)))[0]  
                        V = cv2.VideoCapture(videoFile)
                        V.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp * 1e3)
                        _, frame = V.read()
                        h = numpy.size(frame, 0)
                        w = numpy.size(frame, 1)
                        x1 = int(row['entity_box_x1'] * w)
                        y1 = int(row['entity_box_y1'] * h)
                        x2 = int(row['entity_box_x2'] * w)
                        y2 = int(row['entity_box_y2'] * h)
                        face = frame[y1:y2, x1:x2, :]
                        j = j+1
                        imageFilename = os.path.join(insDir, str(count)+'.jpg')
                        count+=1
                        entityID
                        result = cv2.imwrite(imageFilename, face)
                        if result == False:
                            print('ERROR AT : '+str(count)+'.jpg')
                        entityID.append(int(row['entity_id'][-1])-1)
                
                unique_index = numpy.unique(frame_index)
                # index_to_int = {index: i for i, index in enumerate(unique_index)}
                # frame_index = [index_to_int[index] for index in frame_index]            
                datalist.append([video_id, num_frames, frame_index, entityID, pos, labels])
                print(f'Id: {video_id}, Index max is :{max(frame_index)}, totalframe is: {num_frames}')
                break

        # with open(f'{dataType}.pkl', 'wb') as file:
        #     pickle.dump(datalist, file)

def extract_target_clips(dataset_sets, csv_dir, clip_vid_dir, orig_vid_dir, extract_body_data=False):
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    header = ['videoName', 'entityID', 'fps', 'numFrames', 'facePos', 'labels']
    for dataType in dataset_sets:
        print("Extracting {} data of {} set...".format('face', dataType))
        csv_file = "{}_orig.csv".format(dataType)

        df = pandas.read_csv(os.path.join(csv_dir, csv_file))
                
        dfNeg = df[df['label_id'] == 0]
        dfPos = df[df['label_id'] == 1]

        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')

        outDir = os.path.join(clip_vid_dir, dataType)

        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)


        for entity in tqdm.tqdm(entityList, total = len(entityList)):

            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            if videoKey == "0dY0t1NRXeo_275-305":
                entityID = insData.iloc[0]['entity_id']
                videoDir = os.path.join(orig_vid_dir, dic[dataType])
                videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
                V = cv2.VideoCapture(videoFile)
                
                # only store entityID
                insDir = os.path.join(os.path.join(outDir, videoKey, entityID[-1]))

                if not os.path.isdir(insDir):
                    os.makedirs(insDir)
                j = 0

                for _, row in insData.iterrows():
                    imageFilename = os.path.join(insDir, str("%.4f"%row['frame_timestamp'])+'.jpg')
                    if os.path.exists(imageFilename):
                        # print('skip', image_filename)
                        continue
                    V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
                    _, frame = V.read()
                    h = numpy.size(frame, 0)
                    w = numpy.size(frame, 1)
                    x1 = int(row['entity_box_x1'] * w)
                    y1 = int(row['entity_box_y1'] * h)
                    x2 = int(row['entity_box_x2'] * w)
                    y2 = int(row['entity_box_y2'] * h)
                    face = frame[y1:y2, x1:x2, :]
                    j = j+1
                    cv2.imwrite(imageFilename, face)

if  __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--body", action='store_true', help='Extract body data')
    args = argParser.parse_args()
    extract_body_data = args.body

    WASD_dir = "WASD"
    orig_vids = "orig_videos"
    orig_audios = "orig_audios"
    cvs_dir     = "csv"
    clip_audios_dir = "clips_audios"
    clip_videos_dir = "clips_videos"
    dataset_sets = ['train', 'val'] # ['val']

    orig_audios_fullpath  = os.path.join(WASD_dir, orig_audios)
    orig_vids_fullpath = os.path.join(WASD_dir, orig_vids)


    # print("####### EXTRACTING AUDIO #######")

    # extract_audio(orig_vids_fullpath, orig_audios_fullpath)

    csv_dir_fullpath  = os.path.join(WASD_dir, cvs_dir)
    clip_audios_dir_fullpath = os.path.join(WASD_dir, clip_audios_dir)

    # print("####### STARTING AUDIO SLICING #######")

    #extract_audio_clips(dataset_sets, csv_dir_fullpath, clip_audios_dir_fullpath, orig_audios_fullpath)

    # print("####### FINISHED AUDIO SLICING #######")

    clip_videos_dir_fullpath    = os.path.join(WASD_dir, clip_videos_dir)

    # if extract_body_data:
    #     print("####### STARTING FACE AND BODY CROPPING #######")
    # else:
    #     print("####### STARTING FACE CROPPING #######")
    write_annotation(dataset_sets, csv_dir_fullpath, orig_vids_fullpath)
    # extract_target_clips(dataset_sets, csv_dir_fullpath, clip_videos_dir_fullpath, orig_vids_fullpath, extract_body_data)    
    #extract_video_clips(dataset_sets, csv_dir_fullpath, clip_videos_dir_fullpath, orig_vids_fullpath, extract_body_data)

    # if extract_body_data:
    #     print("####### FINISHED FACE AND BODY CROPPING #######")
    # else:
    #     print("####### FINISHED FACE CROPPING #######")

