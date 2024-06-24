# """2023 / 7 / 16
# 11: 46
# dell
# 箱线图绘制
# """
# libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
from matplotlib.font_manager import FontProperties
confidence_level = 0.95
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
sns.set(style="ticks")
# 加载示例数据集

# df = pd.read_excel(r'G:\IPMI_Jornal\BRODMANN_DATA\23_hu_ma\eval\RSA_result\cognition2.xlsx', sheet_name='Sheet1')  #BA23
# df = pd.read_excel(r'G:\IPMI_Jornal\BRODMANN_DATA\28_transfer\eval\RSA_result\cognition2_1.xlsx', sheet_name='Sheet1')  #BA23
# df = pd.read_excel(r'G:\IPMI_Jornal\CHARM_DATA\eval\RSA_result\cognition2_1.xlsx', sheet_name='Sheet1')  #BA23
df = pd.read_excel(r'C:\Users\dell\Desktop\rsa_avg.xlsx', sheet_name='Sheet1')  #cvae
# df = sns.load_dataset('tips')
df.head()


al = {}
group_data = {}
# group_data1 = {}
hue_groups = df.groupby('type')
for hue_group_name, hue_group_data in hue_groups:
    a = hue_group_data.groupby('Behavior')

    # print(f'Hue Group: {hue_group_name}')
    # print(hue_group_data)
    for hue_group_name1, hue_group_data1 in a:

        # print(hue_group_name1)
        # print(al.keys())
        # print(hue_group_name,hue_group_name1,hue_group_data1['Kendall-τ'].mean(),hue_group_data1['Kendall-τ'].std())
        if hue_group_name1 not in al:
            # print(hue_group_data1['type'].to_numpy()[0])
            # if hue_group_data1['type'].to_numpy()[0]== "Human-Specific":
            group_data[hue_group_name1] = [hue_group_data1['Kendall-τ'].to_numpy().tolist()]
            # else:
            #     group_data1[hue_group_name1] = hue_group_data1['Kendall-τ']
            al[hue_group_name1] = [[hue_group_name,hue_group_data1['Kendall-τ'].mean(),hue_group_data1['Kendall-τ'].std()]]
        else:
            al[hue_group_name1].append([hue_group_name, hue_group_data1['Kendall-τ'].mean(),
                                   hue_group_data1['Kendall-τ'].std()])
            # if hue_group_data1['type'].to_numpy()[0] == "Human-Specific":
            group_data[hue_group_name1].append(hue_group_data1['Kendall-τ'].to_numpy().tolist())
            # else:
            #     group_data1[hue_group_name1].append(hue_group_data1['Kendall-τ'])


# for k in group_data.keys():
#     print(k)
#     print(group_data[k])
#
# # for k in group_data1.keys():
# #     print(k)
# #     print(group_data1[k])

font = {'family': 'Times New Roman'}
plt.rc('font', **font)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

colors = ['#00ff00','#008b4f']

all = sns.boxplot(x="Behavior", y="Kendall-τ", hue="type", data=df, palette=colors, width=0.6,showcaps=False,
            showmeans=True,showfliers=False,medianprops = dict(linewidth=0),whiskerprops=dict(linewidth=0),
            meanline=True,meanprops={'linestyle':'-','linewidth':1.5,'color':'white'},autorange=True,boxprops = dict(linewidth=0),
            )
# sns.stripplot(x='day', y='total_bill', data=df, color="black", zorder=1,jitter=0.2, size=4)
ax = sns.swarmplot(x='Behavior', y='Kendall-τ', hue="type",data=df,   size=3,dodge=True,legend=False,
                   palette=['black','black'],alpha =0.6)
ax.set_xlabel('', fontsize=15)
ax.set_ylabel('Kendall-τ', fontsize=16)
# plt.legend(loc='best')
ax.legend(loc='center left', bbox_to_anchor=(0.8, 1.05),prop=FontProperties(weight='bold',size=14))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.xticks(rotation=10)

x_order =  [tick for tick in plt.xticks()]
order_x = [x_order[1][i].get_text() for i in range(len(x_order[1]))]
print(order_x)
# print(x_order[1][0].get_text())


# degrees_of_freedom = 6-1
# t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

for i in range(len(al)):
    if i==0:
        w2 = (al[order_x[i]][0][1]-0.008436857,al[order_x[i]][0][1]+0.008436857)
        w1 = (al[order_x[i]][1][1]-0.005253306,al[order_x[i]][1][1]+0.005253306)


        degrees_of_freedom = 9 - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error1 = t_critical * (al[order_x[i]][0][2] / np.sqrt(9))
        w2 = (al[order_x[i]][0][1] - margin_of_error1, al[order_x[i]][0][1] + margin_of_error1)
        margin_of_error2 = t_critical * (al[order_x[i]][1][2] / np.sqrt(9))
        w1 = (al[order_x[i]][1][1] - margin_of_error2, al[order_x[i]][1][1] + margin_of_error2)
        print(w1,w2)
    elif i==1:
        w2 = (al[order_x[i]][0][1] - 0.021152171,al[order_x[i]][0][1] + 0.021152171)

        w1 = (al[order_x[i]][1][1]-0.002702879,al[order_x[i]][1][1]+0.002702879)

        degrees_of_freedom = 4 - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error1 = t_critical * (al[order_x[i]][0][2] / np.sqrt(4))
        w2 = (al[order_x[i]][0][1] - margin_of_error1, al[order_x[i]][0][1] + margin_of_error1)
        margin_of_error2 = t_critical * (al[order_x[i]][1][2] / np.sqrt(4))
        w1 = (al[order_x[i]][1][1] - margin_of_error2, al[order_x[i]][1][1] + margin_of_error2)
        print(w1, w2)
    else:
        w2 = (al[order_x[i]][0][1] -0.019768744,al[order_x[i]][0][1] +0.019768744)
        w1 = (al[order_x[i]][1][1] - 0.003010399,al[order_x[i]][1][1] + 0.003010399)

        degrees_of_freedom = 4 - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error1 = t_critical * (al[order_x[i]][0][2] / np.sqrt(4))
        w2 = (al[order_x[i]][0][1] - margin_of_error1, al[order_x[i]][0][1] + margin_of_error1)
        margin_of_error2 = t_critical * (al[order_x[i]][1][2] / np.sqrt(4))
        w1 = (al[order_x[i]][1][1] - margin_of_error2, al[order_x[i]][1][1] + margin_of_error2)
        print(w1, w2)


    z1 = [i-0.15,i-0.15]
    z2 = [i + 0.15, i + 0.15]
    z3 = [i , i ]





    _, p_value = stats.ttest_rel(group_data[order_x[i]][1],group_data[order_x[i]][0])
    print(al[order_x[i]][1][1], al[order_x[i]][0][1], p_value)
    # print(p_value)
    if p_value<0.0001:
        ax.text(i, ax.get_ylim()[0], '****', color='red', fontsize=14, ha='center', va='center')
        # plt.plot(i, 0, marker='****', color='red')
    elif p_value<0.001:
        ax.text(i, ax.get_ylim()[0], '***', color='red', fontsize=14, ha='center', va='center')
    elif p_value<0.01:
        ax.text(i, ax.get_ylim()[0], '**', color='red', fontsize=14, ha='center', va='center')
    elif p_value<0.05:
        ax.text(i, ax.get_ylim()[0], '*', color='red', fontsize=14, ha='center', va='center')




    # w1 = [al[order_x[i]][0][1] - al[order_x[i]][0][2],al[order_x[i]][0][1] + al[order_x[i]][0][2]]
    # w2 = [al[order_x[i]][1][1] - al[order_x[i]][1][2], al[order_x[i]][1][1] + al[order_x[i]][1][2]]
    plt.plot(z1,w1,color='#FF6100',linewidth=1.5)
    plt.plot(z2, w2, color='#FF6100',linewidth=1.5)


# plt.title("Boxplot with jitter", loc="center")
# print(al)
plt.show()

