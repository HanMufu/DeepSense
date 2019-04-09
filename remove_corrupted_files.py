import os

dataDir = 'variable_length_training_samples_512'

sample_dir = dataDir
sample_list = os.listdir(sample_dir)
for sample_name in sample_list:
    file_in = open(os.path.join(sample_dir, sample_name))
    try:
        sample = file_in.readline()[:-1].split(',')
    except:
        print(os.path.join(sample_dir, sample_name))
        os.remove(os.path.join(sample_dir, sample_name))
    else:
        continue