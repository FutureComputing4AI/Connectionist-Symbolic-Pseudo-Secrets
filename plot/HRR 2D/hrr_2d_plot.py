import tikzplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as f

sns.set_style('darkgrid')
plt.figure(figsize=[10, 8])

# loading data
pos_idx = np.load('data/pos_idx.npy')
neg_idx = np.load('data/neg_idx.npy')
proj_pos_idx = np.load('data/proj_pos_idx.npy')
proj_neg_idx = np.load('data/proj_neg_idx.npy')
idx = np.arange(1, len(pos_idx) + 1)

# loading font
font = f.FontEntry(fname='./../fonts/Lato.ttf', name='lato')
f.fontManager.ttflist.insert(0, font)

# setting text font
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['mathtext.fontset'] = 'cm'

# print(plt.rcParams.keys())
# plotting
linewidth = 2.5
plt.plot(idx, pos_idx, color='#8142ff', label='Naive Present', linewidth=linewidth)
plt.plot(idx, proj_pos_idx, color='#23f011', label='Proj Present', linewidth=linewidth)
plt.plot(idx, neg_idx, color='#ff8e42', label='Naive Absent', linewidth=linewidth)
plt.plot(idx, proj_neg_idx, color='#e32bff', label='Proj Absent', linewidth=linewidth)

plt.ylim([-1.5, 2])
plt.legend(loc='lower right', borderpad=1., handlelength=2.5, fancybox=True, framealpha=0.5)
plt.xlabel('Number of bound terms')
plt.ylabel('$\sum_{i} \;\; x_i \cdot (S \; \circledast \; y_{i}^{\dagger})$', fontsize=24, fontweight='heavy')
plt.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.98, wspace=0, hspace=0)
plt.savefig('./../../figure/HRR 2D.png', bbox_inches="tight", pad_inches=0)
plt.savefig('./../../figure/HRR 2D.pdf', bbox_inches="tight", pad_inches=0)
tikzplotlib.save("./../../figure/HRR 2D.tex")
plt.show()
