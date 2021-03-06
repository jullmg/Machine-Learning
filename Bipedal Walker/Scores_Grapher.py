import numpy as np
import matplotlib.pyplot as plt


suffix = '02'
suffix_2 = '02'

scores_save_name = './Bipedal_Logs/Score-{}.npy'.format(suffix)
scores_2_save_name = './Bipedal_Logs/Score-{}.npy'.format(suffix_2)


scores = np.load(scores_save_name)
scores_2 = np.load(scores_2_save_name)

plt.plot(scores)
# plt.plot(scores_2)
# plt.legend(scores)
plt.ylim(-250, 325)
plt.title(scores_save_name)

plt.show()
