## Instructions

1. Change config file (change dataset_dir, output_dir, modify numbers if necessary)
2. Make sure you are running all the files in the data_selection directory

### Step 1: Sample dataset
3. Run sample_dataset.py, which will generate a test.csv containing the labels for all the files
4. It will also create a new folder called to_augment, which will contain the videos to be augmented in the next step

### Step 2: Augment dataset
5. Run augment_dataset.py (make sure you have already installed TransferAttack on anaconda), which will process the videos in to_augment and augment it into the folder augmented

Note that this can be used to augment individual video files by adding them to the folder to_augment and running augment_dataset.py

### Step 3: Collate and Compress dataset
6. Run collate_and_compress_dataset.py, which will copy and compress everything into the collated folder


