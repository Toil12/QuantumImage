import numpy as np
name1="angle_20-000859_loss.npy"
name2="FRQI_18-103207_model.npy"
path1=f"data/loss_epoch/{name1}"
path2=f"data/model/{name2}"

a=np.load(path1)
print(a)