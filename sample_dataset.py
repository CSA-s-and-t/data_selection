import os 
import shutil
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

NA_CHARACTER = '-'

'''
The goal of this script is to perform the random sampling of the videos based on our configuration settings (config.yaml), saving all the data
into a collated CSV file (with paths to each file) test.csv
'''

main_df = pd.DataFrame(columns=['original_path', 'label', 'target', 'technique', 'source'])

def add_to_df(paths: list[str], label, source, target = NA_CHARACTER, technique = NA_CHARACTER):
    global main_df
    temp_df = pd.DataFrame()
    temp_df['original_path'] = paths
    temp_df['label'] = label
    temp_df['technique'] = technique
    temp_df['target'] = target
    temp_df['source'] = source
    main_df = pd.concat([main_df, temp_df], ignore_index=True)

def even_divide(num, div):
    group_size, remainder = divmod(num, div)
    return [group_size + (1 if x < remainder else 0) for x in range(div)]

def get_videos(original_dataset_paths, manipulated_dataset_paths, num_original, num_manipulated, dataset_name, dataset_technique):
    assert num_manipulated == num_original

    # Step 1: Get original videos
    for path in original_dataset_paths:
        if not os.path.exists(path):
            print(f"No path {path} detected")

        # for celebdf, some original videos do not have corresponding deepfakes. need to remove them from the sampled list
        if dataset_name == 'CelebDF':
            video_list = list(set([os.path.join(path, f.split('_')[-3] + '_' + f.split('_')[-1]) for f in os.listdir(manipulated_dataset_paths[0]) 
                                   if os.path.isfile(os.path.join(manipulated_dataset_paths[0], f))]))
            
        # get list of video paths in folder without visiting subdirectories
        else:
            video_list =  [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # sample num_original videos from this list and add to dataframe
        video_list = random.sample(video_list, num_original)
        add_to_df(video_list, 0, dataset_name)
    
    # Step 2: Get list of manipulated videos
    video_list_manip = []
    sample_sizes = even_divide(num_manipulated, len(manipulated_dataset_paths))

    for path in manipulated_dataset_paths:
        if not os.path.exists(path):
            print(f"No path {path} detected")

        # get list of video paths in folder without visiting subdirectories
        video_list_manip.append([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    
    # Step 3: Pair original videos with manipulated videos

    #DFD is fundamentally unpaired
    if dataset_name == 'DFD':
        add_to_df(random.sample(video_list_manip[0], num_manipulated), 1, dataset_name, '-', dataset_technique)
        return
        
    for orig_video in video_list: # orig_video is path to orignial video
        orig_video_name = Path(orig_video).name
        for i in range(len(sample_sizes)):
            if sample_sizes[i] == 0:
                continue

            # if no default technique is specified (ff++), get technique from path name
            if dataset_name == 'FaceForensics++':
                fake_video = [video for video in video_list_manip[i] if (orig_video_name.removesuffix('.mp4') + '_') in video]
                if len(fake_video) > 0:
                    if 'Deepfakes' in manipulated_dataset_paths[i]._str or 'FaceSwap' in manipulated_dataset_paths[i]._str:
                        dataset_technique = 'faceswap'
                    elif 'NeuralTextures' in manipulated_dataset_paths[i]._str or 'Face2Face' in manipulated_dataset_paths[i]._str:
                        dataset_technique = 'facemanip'
                    add_to_df(random.sample(fake_video, 1), 1, dataset_name, orig_video, dataset_technique)
                    sample_sizes[i] -= 1
                    break
            
            elif dataset_name == 'CelebDF':
                fake_video = [video for video in video_list_manip[i] if orig_video_name in (video.split('_')[-3] + '_' + video.split('_')[-1])]
                assert len(fake_video) > 0
                add_to_df(random.sample(fake_video, 1), 1, dataset_name, orig_video, dataset_technique)
            
            elif dataset_name == 'SelfGenerated':
                fake_video = os.path.join(manipulated_dataset_paths[i], orig_video_name.replace('real', 'fake'))
                assert os.path.isfile(fake_video)
                add_to_df([fake_video], 1, dataset_name, orig_video, dataset_technique)


def get_videos_label(dataset_paths, root_path, num_original, num_manipulated, label_file, dataset_name, dataset_technique):
    assert num_manipulated == num_original
    if not os.path.exists(label_file):
        print(f"No path {label_file} detected")
        return
    
    df = pd.read_csv(label_file)
    # Processing FakeAVCeleb: first take n reals. then, from the pool of reals, take faceswap / fsgan / wav2lip from FakeVideo real audio, and faceswap / fsgan / wav2lip from fake video fake audio
    if 'race' in df: 
        original_df = df[df['method'] == 'real'].sample(num_original)
        video_list = [os.path.join(root_path, item) for item in original_df['path'].str[12:] + '/' + original_df['name']]
        add_to_df(video_list, 0, dataset_name)

        types = ['FakeVideo-RealAudio', 'FakeVideo-RealAudio', 'FakeVideo-RealAudio', 'FakeVideo-FakeAudio', 'FakeVideo-FakeAudio', 'FakeVideo-FakeAudio']
        methods = ['faceswap', 'fsgan', 'wav2lip', 'faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip']
        techniques = ['faceswap', 'faceswap', 'lipsync', 'faceswap-lipsync-voiceclone', 'faceswap-lipsync-voiceclone', 'lipsync-voiceclone']
        sample_sizes = even_divide(num_manipulated, len(methods))
        for _, row in original_df.iterrows():
            for i in range(len(sample_sizes)):
                if sample_sizes[i] == 0:
                    continue
                temp_df = df[(df['source'] == row['source']) & (df['type'] == types[i]) & (df['method'] == methods[i])]
                if len(temp_df) > 0:
                    sample_sizes[i] -= 1
                    temp_df = temp_df.sample(1).iloc[0]
                    add_to_df([os.path.join(root_path, temp_df['path'][12:] + '/' + temp_df['name'])], 1, dataset_name, 
                              os.path.join(root_path, row['path'][12:] + '/' + row['name']), techniques[i])
                    break
    
    # Processing DFDC: no matching of reals and fakes. sample from the label file, pool all the folders, and find it in the folder listing
    else:
        video_list = []        
        original_df = df[df['label'] == 0].sample(num_original)
        for file in original_df['filename']:
            for path in dataset_paths:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    video_list.append(file_path)
                    break
        add_to_df(video_list, 0, dataset_name)

        video_list = []        
        manipulated_df = df[df['label'] == 1].sample(num_manipulated)
        for file in manipulated_df['filename']:
            for path in dataset_paths:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    video_list.append(file_path)
                    break        
        add_to_df(video_list, 1, dataset_name, '-', dataset_technique)


def main():
    # from config.yaml load parameters
    yaml_path = './config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    # Get the parameters
    dataset_names = config['processed_datasets']
    dataset_root_path = config['dataset_dir']
    dataset_metadata = config['dataset_info']

    for dataset_name in dataset_names:
        if dataset_name not in dataset_metadata:
            continue
        dataset_info = dataset_metadata[dataset_name]
        dataset_path = Path(os.path.join(dataset_root_path, dataset_info['folder']))
        num_original = dataset_info['original']
        num_manipulated = dataset_info['manipulated']

        if 'label_file' in dataset_info:
            label_path = Path(os.path.join(dataset_path, dataset_info['label_file']))
        else:
            label_path = None
        dataset_technique = dataset_info['technique'] if 'technique' in dataset_info else None

        if dataset_name == 'CelebDF':
            original_dataset_names = ['Celeb-real']
            manipulated_dataset_names = ['Celeb-synthesis']
            original_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in original_dataset_names]
            manipulated_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in manipulated_dataset_names]

        elif dataset_name == 'DFD':
            original_dataset_names = ["original_sequences/actors/c23/videos"]
            manipulated_dataset_names = ["manipulated_sequences/DeepFakeDetection/c23/videos"]
            original_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in original_dataset_names]
            manipulated_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in manipulated_dataset_names]

        elif dataset_name == 'FaceForensics++':
            original_dataset_names = ["original_sequences/youtube/c23/videos"]
            manipulated_dataset_names = [ "manipulated_sequences/Deepfakes/c23/videos", \
                            "manipulated_sequences/Face2Face/c23/videos", "manipulated_sequences/FaceSwap/c23/videos", \
                            "manipulated_sequences/NeuralTextures/c23/videos"]
            original_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in original_dataset_names]
            manipulated_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in manipulated_dataset_names]
        
        elif dataset_name == 'DFDC': # need to refer to labels, just pool them all together
            sub_dataset_names = ['0', '1', '2', '3', '4']
            sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

        elif dataset_name == 'FakeAVCeleb': # handled using label file
            sub_dataset_paths = None

        elif dataset_name == 'SelfGenerated':
            original_dataset_names = ['original']
            manipulated_dataset_names = ['manipulated']
            original_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in original_dataset_names]
            manipulated_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in manipulated_dataset_names]
        
        if not label_path:
            get_videos(original_dataset_paths, manipulated_dataset_paths, num_original, num_manipulated, dataset_name, dataset_technique)
        else:
            get_videos_label(sub_dataset_paths, dataset_path, num_original, num_manipulated, label_path, dataset_name, dataset_technique)
        

    augmentation_techniques = config['augmentation_techniques']

    # sample 15 videos to use for augmentation, 5 from fakeavceleb, 5 from celeb-df-v2, all from self generated
    N_SAMPLE = 5
    augmented_df = pd.concat([main_df[(main_df['source'] == 'CelebDF') & (main_df['label'] == 1)].sample(N_SAMPLE), 
                              main_df[(main_df['source'] == 'FakeAVCeleb') & (main_df['technique'].str.contains('lipsync'))].sample(N_SAMPLE),
                              main_df[(main_df['source'] == 'SelfGenerated') & (main_df['label'] == 1)].sample(N_SAMPLE)])
    augmented_df["target"] = '-'

    for index, row in augmented_df.iterrows():
        #transfer videos to new folder to_augment for next step: augmentation
        src_file = row["original_path"]
        dst_folder = os.path.join(dataset_root_path, 'to_augment')
        new_path = shutil.copy(src_file, dst_folder)
        augmented_df.at[index, "target"] = row["original_path"]
        augmented_df.at[index, "original_path"] = new_path.replace('to_augment', 'augmented')
    
    # add augmentations
    tiling = np.tile(augmentation_techniques, len(augmented_df))
    augmented_df = augmented_df.loc[augmented_df.index.repeat(len(augmentation_techniques))].reset_index(drop=True)
    augmented_df['augmentation'] = tiling
    augmented_df['original_path'] = augmented_df['original_path'].str.removesuffix('.mp4') + '_' + augmented_df['augmentation'] + '.mp4'

    main_df['augmentation'] = '-'
    main_df['tool'] = '-'
    main_df['content'] = '-'
    pd.concat([main_df, augmented_df], ignore_index=True).to_csv('test.csv', index=False) 

if __name__ == "__main__":
    main()
