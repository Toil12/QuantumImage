import numpy as np
name1="FRQI_28-135437_loss.npy"
name2="FRQI_28-135437_model.npy"
path1=f"data/loss_epoch/{name1}"
path2=f"data/model/{name2}"

a=np.load(path1)
print(a)