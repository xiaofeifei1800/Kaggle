import os

#################
## Preprocesss ##
#################
#### preprocess data
cmd = "python ./clean_data.py"
os.system(cmd)

#######################
## Generate features ##
#######################
#### basic feature
cmd = "python ./base_feature.py"
os.system(cmd)

cmd = "python ./base_feature1.py"
os.system(cmd)

#### count feature
cmd = "python ./count.py"
os.system(cmd)

#### distance feature
cmd = "python ./distance.py"
os.system(cmd)

#### magic feature
cmd = "python ./magic_feature.py"
os.system(cmd)

### russian feature
cmd = "python ./russian feature.py"
os.system(cmd)

### merge all feature
cmd = "python ./merge.py"
os.system(cmd)