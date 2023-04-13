#! /usr/bin/env python2

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


workload1 = np.array([47, 44.56, 66.78, 30.11, 40.67, 45.89, 58.78, 50.78, 33.33, 40.44, 24.33, 33.78, 74.44, 84.89])
workload2 = np.array([37.22, 40.89, 25.89, 32.44, 36, 7.56, 40.22, 51.44, 29.44, 44.44, 34.22, 46.67, 10.11, 37.78])
usability1 = np.array([83.3333333333333, 71.2121212121212, 53.030303030303, 71.2121212121212, 57.5757575757576,
                       72.7272727272727, 51.5151515151515, 68.1818181818182, 62.1212121212121, 59.0909090909091,
                       83.3333333333333, 60.6060606060606, 33.33333333, 43.93939394])
usability2 = np.array([66.6666666666667, 53.030303030303, 98.4848484848485, 72.7272727272727, 69.6969696969697,
                       92.4242424242424, 71.2121212121212, 68.1818181818182, 83.33333333, 98.48484848,
                       81.81818182, 93.93939394, 80.3030303, 83.33333333])
sit_awness1 = np.array([30.56, 41.67, 27.78, 30.56, 41.67, 22.22, 36.11, 44.44, 25, 55.56, 22.22, 38.89, 38.89, 27.78])
sit_awness2 = np.array([33.33, 13.89, 25, 19.44, 30.56, 41.67, 19.44, 19.44, 25, 33.33, 22.22, 8.33, 30.56, 30.56])

fig1, ax1 = plt.subplots(1, 3, figsize=(6, 3))
fig1.tight_layout()

ax1[0].boxplot([workload1, workload2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['C1', 'C2'])
ax1[0].set_ylabel('Workload')
ax1[0].set_ylim(0, 99)

ax1[1].boxplot([usability1, usability2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['C1', 'C2'])
ax1[1].set_ylabel('Usability')
ax1[1].set_ylim(0, 99)

ax1[2].boxplot([sit_awness1, sit_awness2], patch_artist=False,
                positions=[1,1.5],
                widths=[0.5, 0.5],
                labels=['C1', 'C2'])
ax1[2].set_ylabel('Situational awareness')
ax1[2].set_ylim(0, 99)

fig1.savefig('/home/i2r2020/Documents/huanfei/bo_data/score_questionanire.tif', dpi=300, bbox_inches="tight")


fig2, ax2 = plt.subplots(1, 3, figsize=(6, 3))
fig2.tight_layout()

collision1 = np.array([3, 4, 3, 5, 3, 6, 4, 9, 5, 5, 5, 8, 7, 6, 5, 4])
collision2 = np.array([6, 0, 0, 1, 3, 0, 0, 2, 1, 2, 2, 6, 3, 3, 5, 3])
contact_lost1 = np.array([6, 6, 5, 7, 6, 7, 6, 9, 6, 5, 6, 7, 7, 6, 3, 6])
contact_lost2 = np.array([4, 1, 1, 2, 4, 1, 1, 2, 3, 4, 2, 3, 3, 6, 5, 3])
completion_time1 = np.array([54, 64, 71, 61, 51, 60, 52, 72, 56, 66, 74, 57, 62, 63, 49, 62])
completion_time2 = np.array([57, 48, 67, 56, 57, 67, 33, 56, 50, 58, 51, 66, 54, 48, 55, 51])

ax2[0].boxplot([collision1, collision2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['C1', 'C2'])
ax2[0].set_ylabel('Collision')
ax2[0].set_ylim(0, 10)
ax2[1].boxplot([contact_lost1, contact_lost2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['C1', 'C2'])
ax2[1].set_ylabel('Contact loss')
ax2[1].set_ylim(0, 10)
ax2[2].boxplot([completion_time1, completion_time2], patch_artist=False, positions=[1, 1.5], widths=[0.5, 0.5], labels=['C1', 'C2'])
ax2[2].set_ylabel('Time (minutes)')
ax2[2].set_ylim(0, 90)
fig2.savefig('/home/i2r2020/Documents/huanfei/bo_data/performance.tif', dpi=300, bbox_inches="tight")

fig3, ax3 = plt.subplots(3,1)
fig3.tight_layout()
chi_square = np.array([[4.02,4.15,28.48,5.36,11.42,3.96,6.7,5.49,3.96,1.47,1.13,5.8,4,3.9,5.28,4.4],
                      [4.03, 4.66, 17.6, 5.32, 5.88, 3.75, 6.9, 2.72, 3.75, 1.64, 1.19, 7, 3.94, 4.03, 4.6, 3.9]])

rmsea = np.array([[0.013130643285972,0.035959747611404,0.459384896093266,0.108278058400742,
                   0.252914051095188,0,0.152564288314682,0.113335023652033,0,0,0,0.12456821978061,0,0,
                   0.105045,0.058722],
                  [0.108807538616449,0.138132037450097,0.409653624363343,0.163299316185545,0.078,0.092847669088526,
                   0.085,0.053,0.063,0,0,0.214422506967559,0.064,0.065,0.135613,0.10171]])

bic = np.array([[1315,681,2354,3406,591,20,510,815,146,2606,605,592,3035,528,1312,876],
                [1265,644,1848,3297,497,10,464,756,156,2505,613,564,3130,524,1234,796]])

mase = np.array([0.5084, 0.639, 0.7429, 0.6795, 0.6361, 0.769, 0.6226, 0.7035, 0.5877, 0.9199, 1.144, 0.5652, 0.6074,
                 0.9322, 0.8301, 0.5956])

ax3[0].bar(range(1,17),chi_square[0], width=0.4)
ax3[0].bar(np.array(range(1,17))+0.4, chi_square[1], width=0.4)
ax3[0].set_ylabel(r'$\chi^2$')
ax3[0].set_xlabel('Participant')

ax3[1].bar(range(1,17), rmsea[0], width=0.4)
ax3[1].bar(np.array(range(1,17))+0.4, rmsea[1], width=0.4)
ax3[1].set_ylabel('RMSEA')
ax3[1].set_xlabel('Participant')

ax3[2].bar(range(1,17),bic[0], width=0.4)
ax3[2].bar(np.array(range(1,17))+0.4, bic[1], width=0.4)
ax3[2].set_ylabel('BIC')
ax3[2].set_xlabel('Participant')

fig3.savefig('/home/i2r2020/Documents/huanfei/bo_data/model_fit.tif', dpi=300, bbox_inches="tight")

fig4, ax4 = plt.subplots(3, 1)
fig4.tight_layout()

# ranks = np.array([8,6,16,15,12,3.5,7,9,-3.5,14,-2,5,-13,1,10,11])
ranks = bic[0] - bic[1]
print ranks
ax4[0].bar(range(1, 17), ranks, width=0.4)
ax4[0].set_ylabel('BIC difference')
ax4[0].set_xlabel('Participant')

ax4[1].bar(range(1, 17), mase, width=0.4)
ax4[1].set_ylabel('MASE')
ax4[1].set_xlabel('Participant')

fig4.savefig('/home/i2r2020/Documents/huanfei/bo_data/signed_rank.tif', dpi=300, bbox_inches="tight")
plt.show()