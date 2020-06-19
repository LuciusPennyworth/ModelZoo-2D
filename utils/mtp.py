"""
Created on 2020/6/11 19:12 周四
@author: Matt zhuhan1401@126.com
Description: description
"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial']
import numpy as np

fig=plt.figure(figsize=(5.18,2.6))  # 设置画布大小

x=np.arange(22)  # 柱状图在横坐标上的位置
y1=np.array([89.87, 83.94, 84.70, 77.06, 82.41, 80.30, 85.48, 83.23, 80.86, 77.65, 71.58, 80.32, 72.92, 83.28, 75.65, 73.03, 80.77, 76.87, 61.91, 86.62, 90.80, 88.58])
y2=np.array([93.36, 88.10, 88.98, 83.09, 85.37, 84.51, 88.94, 87.71, 87.27, 83.62, 76.11, 84.31, 76.47, 87.78, 78.74, 77.06, 85.76, 80.99, 63.91, 92.64, 93.74, 92.93])
y3=np.array([86.83, 82.06, 84.13, 77.09, 80.86, 79.55, 84.00, 82.89, 82.45, 78.97, 73.13, 79.83, 73.36, 83.79, 76.16, 73.69, 80.91, 77.12, 62.82, 86.82, 88.18, 87.54])

bar_width=0.24  # 设置柱状图的宽度
# 设置x轴显示内容
test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
tick_label=['scan{}'.format(test_set[i]) for i in range(22)]

# 绘制并列柱状图 注意x轴的设置 颜色设置好后带点透明度会好看些
plt.bar(x,y1,bar_width,color="#E66B1A",alpha=0.4,label='Stage1')
plt.bar(x+bar_width,y2,bar_width,color='#0997F7',alpha=0.4,label='Stage2')
plt.bar(x+bar_width*2,y3,bar_width,color='#DC143C',alpha=0.5,label='Stage3')

ax=fig.add_subplot(111)
# 把图例在图表外分3列显示 loc可以用数字表示位置  frameon为false表示去掉图例的外框
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=3, borderaxespad=0., frameon=False, fontsize=8)  # 显示图例，即label
plt.yticks(range(0,100,10))  # 设置y轴范围
plt.xticks(fontsize=8)  # 图中的数字or字母 包括x轴y轴和图例的字体都要调整到一致
plt.yticks(fontsize=8)
plt.xticks(x+bar_width,tick_label)  # 设置x轴显示内容
ax.set_xticklabels(tick_label, rotation = 45)  # 对x轴的内容旋转45度

# 给y轴的数字带上百分号
import matplotlib.ticker as mtick
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)

plt.show()
# 保存为eps格式(但不支持透明度)
fig.savefig('fig4.eps')

