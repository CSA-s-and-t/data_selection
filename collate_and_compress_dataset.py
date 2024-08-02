import os
import shutil
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

# Step 1: Check that all files are available
# Step 2: Make entries for compressed files
# Step 3: Shuffle and rename files
# Step 4: Move (and compress if necessary) files to destination folder

ffmpeg_compress = 'ffmpeg -i "{input_video_file}" -vcodec libx264 -y -crf {rate_factor} {output_video_file}'

original_to_new = {}

def main():
    # from config.yaml load parameters
    yaml_path = './config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    output_dir = Path(config['output_dir'])

    # open the csv file 
    main_df = pd.read_csv('test.csv')

    # verify if every file exists
    all_found = True
    for _, row in main_df.iterrows():
        if not os.path.isfile(row['original_path']):
            all_found = False
            print(f"Missing file {row['original_path']}")
    
    if not all_found:
        print("Not all files found, exiting")
        #return
    
    # create compressed df
    compressed_df = main_df.copy()
    compressed_df['target'] = compressed_df['original_path']
    indices = random.sample(range(len(compressed_df)), len(compressed_df)//2)
    compressed_df['compression'] = 'c23'
    compressed_df.loc[indices,'compression'] = 'c40'
    main_df['compression'] = 'original'

    #merge compressed and main df
    main_df = pd.concat([main_df, compressed_df], ignore_index=True)

    # shuffle rows - filename will be based off the index
    main_df = main_df.sample(frac=1).reset_index(drop=True)
    main_df['name'] = main_df.index.to_series().apply(lambda x: f'{x:04d}.mp4')

    # fix paths to point towards new names
    for _, row in main_df.iterrows():
        if row['compression'] == 'original':
            original_to_new[row['original_path']] = row['name']
    
    for index, row in main_df.iterrows():
        if row['target'] != '-':
            main_df.at[index, 'target'] = original_to_new[row['target']]
        
    main_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False,
                                        columns=['name', 'target', 'label', 'technique', 'source', 'compression', 'augmentation', 'tool', 'content'])
    
    main_df.to_csv(os.path.join(output_dir, 'labels_with_original_path.csv'), index=False,
                                        columns=['name', 'target', 'label', 'technique', 'source', 'compression', 'augmentation', 'tool', 'content', 'original_path'])

    #final step of collating files
    os.makedirs(output_dir, exist_ok=True)
    for _, row in main_df.iterrows():
        output_video_file = os.path.join(output_dir, row['name'])
        if row['compression'] == 'c23':
            os.system(ffmpeg_compress.format(input_video_file = row['original_path'], output_video_file = output_video_file, rate_factor = 23))
        elif row['compression'] == 'c40':
            os.system(ffmpeg_compress.format(input_video_file = row['original_path'], output_video_file = output_video_file, rate_factor = 40))
        else:
            try:
                shutil.copy(row['original_path'], output_video_file)
            except Exception as e:
                print(f"missing file {row['original_path']}")
    
if __name__ == "__main__":
    main()
