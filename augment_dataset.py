import os
import subprocess
import yaml
from fractions import Fraction
from pathlib import Path

transfer_attack_directory = '../TransferAttack'
anaconda_directory = 'C:\\Users\\ASUS\\anaconda3\\Scripts\\activate.bat'
base_command = f'cd {transfer_attack_directory} && {anaconda_directory} transferattack && python -u main.py '

augment_to_command = {
    'vmifgsm_3': base_command + '--input_dir {directory} --output_dir {results} --attack vmifgsm --model xception --eps 3 --batchsize 8',
    'vmifgsm_6': base_command + '--input_dir {directory} --output_dir {results} --attack vmifgsm --model xception --eps 6 --batchsize 8',
    'gra_3': base_command + '--input_dir {directory} --output_dir {results} --attack gra --model xception --eps 3 --batchsize 8',
    'gra_6': base_command + '--input_dir {directory} --output_dir {results} --attack gra --model xception --eps 6 --batchsize 8',
    'ncs_3': base_command + ' --input_dir {directory} --output_dir {results} --attack ncs --model xception --eps 3 --batchsize 8',
    'ncs_6': base_command + '--input_dir {directory} --output_dir {results} --attack ncs --model xception --eps 6 --batchsize 8',
    'dem_3': base_command + '--input_dir {directory} --output_dir {results} --attack dem --model xception --eps 3 --batchsize 4',
    'dem_6': base_command + '--input_dir {directory} --output_dir {results} --attack dem --model xception --eps 6 --batchsize 4',
    'decowa_3': base_command + '--input_dir {directory} --output_dir {results} --attack decowa --model xception --eps 3 --batchsize 4',
    'decowa_6': base_command + '--input_dir {directory} --output_dir {results} --attack decowa --model xception --eps 6 --batchsize 4',
}   

ffprobe_command = 'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {input_video_path}'
ffmpeg_command = 'ffmpeg -r {frame_rate} -i {results_path}/%03d.png -crf 0 -y -c:v libx264 -vf "fps={frame_rate}" -pix_fmt yuv420p {output_video_path_audioless}'
ffmpeg_copy_audio_command = 'ffmpeg -i {output_video_path_audioless} -i {input_video_path} -y -c copy -map 0:v:0 -map 1:a:0 -shortest {output_video_path}'

yaml_path = './config.yaml'
try:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
except yaml.parser.ParserError as e:
    print("YAML file parsing error:", e)

#Sets of Attack training 
#techniques = config['augmentation_techniques']
techniques = ['gra_3', 'dem_3', 'ncs_3']

input_video_root = Path(config['dataset_dir']) / 'to_augment' #input files
output_video_root = Path(config['dataset_dir']) / 'augmented' #output files
video_list =  [f for f in os.listdir(input_video_root) if os.path.isfile(os.path.join(input_video_root, f))]

for video in video_list:
    print(f"Augmenting video {video}")
    #if '00015_id01058_2DKbOjf3Wyo_faceswap_id00919_wavtolip' not in video:
        #continue
    #if 'eFC0STvhou4' in video or 'uKVqPfRf-ac' in video or '00066_id03168_wavtolip' in video:
        #continue
    #if 'wavtolip' not in video:
        #continue
    os.system(f'cd {transfer_attack_directory} && {anaconda_directory} transferattack \
               && python -u processVideo.py {input_video_root} {video}')
    for technique in techniques:
        print(f"Performing augmentation {technique} on {video}")
        frames_path = input_video_root / 'frames_aug' / video.removesuffix('.mp4')
        results_path = input_video_root / 'results' / video.removesuffix('.mp4') / technique
        output_video_path_audioless = output_video_root / f"{video.removesuffix('.mp4')}_{technique}_audioless.mp4"
        output_video_path = output_video_root / f"{video.removesuffix('.mp4')}_{technique}.mp4"

        os.makedirs(results_path, exist_ok=True)
        #get base framerate of original video
        frame_rate_str = subprocess.check_output(ffprobe_command.format(input_video_path = os.path.join(input_video_root, video)), shell=True, text=True)
        frame_rate = round(float(Fraction(frame_rate_str)))
        #perform augmentation
        os.system(augment_to_command[technique].format(directory = frames_path, results = results_path))
        #combine frames into video
        os.system(ffmpeg_command.format(results_path = results_path, output_video_path_audioless = output_video_path_audioless, frame_rate = frame_rate))
        #stitch audio to video
        os.system(ffmpeg_copy_audio_command.format(output_video_path_audioless = output_video_path_audioless, 
                                                   input_video_path = os.path.join(input_video_root, video), 
                                                   output_video_path = output_video_path))
        
        if not os.path.isfile(output_video_path) and os.path.isfile(output_video_path_audioless):
            print(f"original {video} has no audio: renaming audioless")
            os.rename(output_video_path_audioless, output_video_path)
