import splitfolders  # or import split_folders

input_folder = r'C:\Users\nika7\machine_learning_exercise\FER_Custom_Dataset'
output=r'C:\Users\nika7\machine_learning_exercise\train_vari_test'
splitfolders.ratio(input_folder, output="output", 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None) # default values