import json
import os
print("Removing files...")

keys_g = []
keys_gr = []
with open("images/gt_reduced.json") as f:
    data = json.load(f).keys()
    keys_gr = list(data)
    f.close()

with open("images/gt.json") as f:
    data = json.load(f)
    keys_train = list(data["train"].keys())
    keys_test = list(data["test"].keys())
    keys_g = list(set(keys_train + keys_test))
    f.close()

keys = list(set(keys_g + keys_gr))

for i in os.listdir("images/train"):
    file_number = str(i)[:-4]
    if(file_number not in keys):
        try:
            os.remove("images/train/"+file_number+".jpg")
            os.remove("images/test/"+file_number+".jpg")
        except FileNotFoundError:
            pass
print("Files from train and test were removed")
