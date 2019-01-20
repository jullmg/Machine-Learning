import numpy as np
import matplotlib.pyplot as plt
import os, re

source = "./LunarLander_Logs/Scores/"
root = os.walk(source)

for root, dir, files in os.walk(source):
    pass

index_list = []

for file in files:
    index = re.search(r"\d\d", file)
    index_list.append(index.group())

print(index_list)

for i in index_list:
    scores_save_name = './LunarLander_Logs/Scores/Score-{}.npy'.format(i)
    print(scores_save_name)
    scores = np.load(scores_save_name)
    plt.plot(scores)


plt.title(scores_save_name)

plt.show()

exit()


# scores_2_save_name = './Bipedal-Hardcore_Logs/Score-{}.npy'.format(suffix_2)


# scores_2 = np.load(scores_2_save_name)


# plt.plot(scores_2)
# plt.legend(scores)

