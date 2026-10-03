"""Descriptive station comparison; no tuning or inferential selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'midas_station_validation_7'/'transfer_v1'
s=pd.read_csv(OUT/'station_duration_scores.csv',dtype={'station_id':str})
station=pd.read_csv(OUT/'station_summary.csv',dtype={'station_id':str})
summary=pd.read_csv(OUT/'group_summary.csv')
grid=pd.read_csv(OUT/'ukcp18_comparable_summary.csv')
duration=pd.read_csv(OUT/'duration_summary_primary.csv')
cohort=pd.read_csv(OUT/'cohort.csv',dtype={'station_id':str})
methods=['UCF','HQT','TPS'];colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
primary=s[s.primary]
leave=[]
for sid in primary.station_id.unique():
    g=primary[primary.station_id!=sid].groupby('method').D_RL.mean()
    leave.append(dict(excluded_station_id=sid,excluded_station_name=primary.loc[primary.station_id==sid,'station_name'].iloc[0],
        best_method=g.idxmin(),**{m:float(g[m]) for m in methods}))
pd.DataFrame(leave).to_csv(OUT/'leave_one_station_out_summary.csv',index=False)
paired=[]
w=primary.pivot(index=['station_id','duration_h'],columns='method',values='D_RL')
for m in ['HQT','TPS']:
    delta=w[m]-w.UCF
    paired.append(dict(comparison=f'{m} minus UCF',mean_difference=delta.mean(),median_difference=delta.median(),
        UCF_lower_pairs=int((delta>0).sum()),pairs=len(delta)))
pd.DataFrame(paired).to_csv(OUT/'paired_method_differences.csv',index=False)

# A provisional discussion figure, not a journal-specific final layout.
order=cohort[cohort.primary].station_id.tolist()+cohort[~cohort.primary].station_id.tolist()
labels=['Stornoway','Heathrow','Boscombe\nDown','Valley','Aberporth','Aldergrove','Boulmer*']
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for col,(metric,title) in enumerate([('D_RL','Absolute log error $D_{RL}$'),('B','Signed log bias $B$'),('S','Shape error $S$')]):
        ax=axes[0,col]
        ax.axvspan(5.5,6.5,color='0.93',zorder=0)
        for offset,(method,color,marker) in enumerate(zip(methods,colors,markers)):
            g=station[station.method==method].set_index('station_id').loc[order]
            ax.scatter(np.arange(7)+(offset-1)*.18,g[metric],c=color,marker=marker,label=method,s=34)
        ax.set_xticks(np.arange(7),labels,rotation=40,ha='right');ax.set_ylabel(title)
        ax.set_title(f'({chr(97+col)}) Station means across 7 durations',loc='left',fontsize=11)
        ax.axhline(0,color='0.5',lw=.7);ax.grid(axis='y',alpha=.18)
        if metric!='B':ax.set_ylim(bottom=0)
        ax=axes[1,col]
        for method,color,marker in zip(methods,colors,markers):
            g=duration[duration.method==method].sort_values('duration_h')
            ax.plot(g.duration_h,g[metric],color=color,marker=marker,label=method,lw=1.8,ms=5)
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,3,4,5,6,12],['1','2','3','4','5','6','12'])
        ax.set_xlabel('Duration (h)');ax.set_ylabel(title)
        ax.set_title(f'({chr(100+col)}) Equal-weight mean of 6 primary stations',loc='left',fontsize=11)
        ax.axhline(0,color='0.5',lw=.7);ax.grid(axis='y',alpha=.18)
        if metric!='B':ax.set_ylim(bottom=0)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    axes[0,2].legend(handles,legend_labels,loc='upper left',frameon=False)
    fig.suptitle('MIDAS historical transfer validation: 1961–1990 → 1991–2020',fontsize=15)
    fig.supxlabel('Point estimates; no confidence intervals. *Boulmer: 14 early-period years; excluded from primary means.',fontsize=10)
    fig.savefig(OUT/'station_transfer_comparison.png',dpi=180)
    fig.savefig(OUT/'station_transfer_comparison.pdf')
    plt.close(fig)
with Image.open(OUT/'station_transfer_comparison.png') as im:
    assert im.width==2340 and im.height==1350
manifest=dict(purpose='Provisional discussion figure',source_tables=['station_summary.csv','duration_summary_primary.csv'],
    aggregation='Equal station weights; seven duration means per station. Boulmer supplementary only.',
    confidence_intervals=False,transforms='Metrics are natural-log errors; duration axis log2',
    alt_text='Six panels compare UCF, HQT and TPS by station and by duration. UCF has the smallest six-station mean absolute log error at every duration. TPS has particularly large error at Stornoway and supplementary Boulmer; HQT has large error at Aberporth.',
    output_size_inches=[13,7.5],png_dpi=180)
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

lines=['# MIDAS站点与UKCP18模式网格：同方法验证结果','',
'## 结论','',
'核心结论一致：UCF的总体平均误差最小；完整方法排名不完全一致。六站主分析为UCF < HQT < TPS，模式陆地网格为UCF < TPS < HQT。不能写成三种方法的观测表现完全复现模式结果。','',
'## 实验口径','',
'使用与模式实验完全相同的平稳L-moment GEV、UCF/HQT/TPS核心计算程序。先用1961—1990小时AMS、前后两段整日1/2/3日AMS冻结预测，再单独读取1991—2020小时AMS拟合评价参照。后段小时雨量值没有参与传递模型拟合、调参或选择；覆盖率和资料可用年份用于事先确定配对样本。','',
'目标历时为1、2、3、4、5、6、12h；输出2、5、10、20、25、50、100年重现水平及平均强度。额外保留20年节点以完全匹配模式程序。前后两段10个小时滑动历时及3个整日历时的GEV参数均保存。窗口结束时间所在自然年口径与旧站点AMS保持一致，与模式Dec—Nov/360日年不相同。','',
'UCF以整日1日重现水平变化因子传递历史小时分位数。HQT使用历史整日CDF与小时分位数映射。TPS分别拟合两段整日GEV强度的位置和尺度幂律，固定各段1日形状参数，将外推变化比作用于历史小时分位数。沿用HQT概率界限[1e-6,1-1e-6]及固定401个logT节点加工程节点的加权log-PAVA。','',
'D_RL为T=2—100上对logT积分的平均绝对对数误差；B为带符号平均对数误差；S为去均值后的对数误差均方根。主积分201点，401点核验。D_RL不是百分比，也不等于B+S。以下均为先逐站/格点计算再等权平均，避免混用均值与中位数。','',
'## 总体对照','',
'|资料|方法|平均D_RL|平均B|平均S|最优比例|','|---|---|---:|---:|---:|---:|']
for label,frame in [('MIDAS六站',summary[summary.group=='primary_6']),('UKCP18四成员陆地网格',grid)]:
    for r in frame.itertuples():
        lines.append(f'|{label}|{r.method}|{r.mean_D_RL:.4f}|{r.mean_B:+.4f}|{r.mean_S:.4f}|{r.best_fraction:.1%}|')
lines+=['','站点最优比例以42个站点—历时组合为分母；网格以四成员×七历时×10397个陆地格点为分母。组合彼此相关，不是独立重复试验。','',
'UCF在六站平均的七个目标历时上均最优。六站中按历时平均误差选方法：Stornoway、Heathrow、Aberporth为UCF；Valley、Aldergrove为HQT；Boscombe Down为TPS。Aldergrove的HQT与UCF很接近（0.07909与0.07957），不应称显著优势。','',
'## 稳健性与差异','',
'纳入补充站Boulmer后，UCF/HQT/TPS的平均D_RL分别为0.1141/0.1487/0.2433，UCF仍最优。逐一移出六个主站，剩余五站的平均D_RL最优方法始终为UCF。剔除三个原有名义趋势站后，仅剩Heathrow、Valley、Aberporth，平均D_RL为0.1419/0.2090/0.2017，UCF仍最优，但此为事后筛选的探索性对照。','',
'B偏差方向不完全一致：六站三方法总体平均B均略为正，而模式UCF为负。站点间正负误差抵消明显，例如TPS在Stornoway明显高估、在若干其他站低估。因此不能据此认为三方法普遍低估，也不能提出统一放大系数。站点的平均S仍为UCF < TPS < HQT，与模式一致，但这不意味着TPS的总误差更小。','',
'## 验证与限制','',
'147条方法—站点—历时评价序列、1029条工程重现水平记录全部有限、正值且重现水平曲线单调。HQT未触发概率裁剪；6条曲线触发了原定单调性修正。GEV分位数已与SciPy交叉检查，误差积分已用独立向量化实现核对；201/401点D_RL最大差约1.03e-5。预测文件在读取后段小时评价资料前冻结并记录哈希，评价后哈希未变。','',
'样本少且各段有效年数不同，尚未计算站点bootstrap区间，因此这里只能说点估计下排名一致，不能说UCF具有统计显著优势。评价“真值”也是后段小时AMS拟合的GEV分位数，并非直接观测到的100年一遇雨量。模式与观测时期、年历、空间支撑及样本长度不同，数值接近不代表模式已被定量验证。','',
'严格主样本为6站；Boulmer因剔除1975年约25%覆盖率记录后前段只剩14年，仅作补充。没有因为趋势显著而删除主样本站点。','',
'## 结果文件','',
'- station_transfer_comparison.png / .pdf：逐站和逐历时D/B/S对照。',
'- group_summary.csv、station_summary.csv、duration_summary_primary.csv：分组、逐站、逐历时汇总。',
'- engineering_return_levels.csv：预测及后段小时GEV参照的雨量、强度、相对误差与变化因子。',
'- prediction_GEV_parameters.csv、evaluation_GEV_parameters.csv：两阶段GEV参数、L-moment、样本年和支持域诊断。',
'- dense_curves.csv：完整201点预测和参照曲线。',
'- paired_method_differences.csv、leave_one_station_out_summary.csv：方法差值及逐站移除诊断。',
'- config.json、prediction_complete.json、FINAL_REPORT.json：配置、输入哈希、冻结与验证记录。','']
(OUT/'站点验证结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
print(pd.DataFrame(leave).to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
print('Figure and discussion report written.')
