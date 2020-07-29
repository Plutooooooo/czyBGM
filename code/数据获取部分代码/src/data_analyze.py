import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *

mpl.rcParams['font.sans-serif']=['SimHei'] # 画图心事中文

classname = ['女装', '手机数码', '日用百货', '男装', '美妆护理', '食品生鲜', 'ALL']
filenamePrefix = "D:\\郭礼华文件\\数据科学\\czyBGM\\data\\"

# 对各个值做一些简单的统计
for i in range(0,len(classname)):
    filename_input = filenamePrefix + classname[i] + '\\' + classname[i] + ".xlsx"
    df = pd.read_excel(filename_input).iloc[:, 3:9]
    print(df.describe())
    filename_output = filenamePrefix + '数据统计_' + classname[i] + '.xlsx'
    df.describe().to_excel(filename_output)

# 画相关性图
for i in range(0,len(classname)):
    filename_input = filenamePrefix + classname[i] + '\\' + classname[i] + ".xlsx"
    df = pd.read_excel(filename_input).iloc[:, 3:8]

    filename_output = filenamePrefix + '相关性分析_' + classname[i] + '.png'
    sns.heatmap(df.corr())
    plt.title(classname[i])

    plt.savefig(filename_output, dpi=200, bbox_inches='tight', title=classname[i])
    plt.show()

# 画单个统计量的分布图
for i in range(0,len(classname)):
    filename_input = filenamePrefix + classname[i] + '\\' + classname[i] + ".xlsx"
    df = pd.read_excel(filename_input).iloc[:, 3:8]

    filename_output = filenamePrefix + '预估销量分布_' + classname[i] + '.png'

    plt.hist(list(df['预估销量']), bins=50, range=[0,100])
    plt.xlabel('预估销量')  # 横轴名
    plt.ylabel('数量')  # 纵轴名
    plt.title(classname[i])
    plt.savefig(filename_output)
    plt.show()
    # sns.distplot(df['预估销量'],hist=True, bins=20,kde=False)

