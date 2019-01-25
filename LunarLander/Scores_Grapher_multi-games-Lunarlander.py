import numpy as np
import matplotlib.pyplot as plt
import os
import re

graph_scores = False


source = "./LunarLander_Logs/Scores/"
root = os.walk(source)

for root, dir, files in os.walk(source):
    pass

scores_list = []
qvalues_list = []

for file in files:

    score_name = re.match(r"Score", file)
    qvalue_name = re.match(r"Qvalues", file)

    if score_name:
        scores_list.append(score_name.string)

    if qvalue_name:
        qvalues_list.append(qvalue_name.string)

if graph_scores:
    for i in scores_list:
        scores_save_name = './LunarLander_Logs/Scores/{}'.format(i)
        print('Loading', scores_save_name)
        scores = np.load(scores_save_name)
        plt.plot(scores)

    plt.title('Scores')
    plt.show()

else:
    for i in qvalues_list:
        qvalue_save_name = './LunarLander_Logs/Scores/{}'.format(i)
        print('Loading', qvalue_save_name)
        qvalues = np.load(qvalue_save_name)
        print(qvalues)
        graph = plt.plot(qvalues)

    plt.title('Target Qvalues')
    plt.show()

# qvalues = np.load()

exit()


# scores_2_save_name = './Bipedal-Hardcore_Logs/Score-{}.npy'.format(suffix_2)


# scores_2 = np.load(scores_2_save_name)


# plt.plot(scores_2)
# plt.legend(scores)

