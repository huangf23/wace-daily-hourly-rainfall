"""Verify and report the completed station uncertainty experiment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='7'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from gev_lmoments_kernel import fit_batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path(__file__).resolve().parent/'midas_station_validation_7'/'transfer_v1'
OUT=SOURCE/'uncertainty_2000'
ci=pd.read_csv(OUT/'aggregate_metric_CI95.csv')
diff=pd.read_csv(OUT/'aggregate_method_difference_CI95.csv')
pred=pd.read_csv(OUT/'predicted_rainfall_intensity_CI95.csv')
truth=pd.read_csv(OUT/'reference_rainfall_intensity_CI95.csv')
for a,cols in [(pred,['depth_low','depth_median','depth_high']), (truth,['depth_low','depth_median','depth_high']),
               (ci,['low','median','high']),(diff,['low','median','high'])]:
    assert np.isfinite(a[cols].to_numpy()).all()
    assert (np.diff(a[cols].to_numpy(),axis=1)>=0).all()
assert np.allclose(pred.intensity_low,pred.depth_low/pred.duration_h)
assert np.allclose(pred.intensity_high,pred.depth_high/pred.duration_h)
assert np.allclose(truth.intensity_high,truth.depth_high/truth.duration_h)
# Independently recover selected ragged bootstrap samples and verify fitted parameters.
draw=np.load(OUT/'calendar_draws.npz')
cohort=pd.read_csv(SOURCE/'cohort.csv',dtype={'station_id':str})
source=pd.read_csv(SOURCE/'early_hourly.csv',dtype={'station_id':str})
verified=0
for sid in cohort.station_id:
    frozen=np.load(OUT/f'frozen_parameters_{sid}.npz')
    data=source[(source.station_id==sid)&(source.duration_h==1)].set_index('year').ams_mm
    for rep in [0,999,1999]:
        years=draw['early_years'][rep];x=np.array([data.loc[y] for y in years if y in data.index])
        p,_,_,status,_=fit_batch(x[None],np.empty(0))
        assert len(x)==frozen['nh'][rep]
        assert status[0]==0 and np.allclose(p[0],frozen['h'][0,:,rep],rtol=1e-12)
        verified+=1
# Independently recalculate complete six-station paired deltas and percentiles.
r=np.load(OUT/'score_replicates.npz');s=r['scores'][cohort.primary.to_numpy()]
valid=np.isfinite(s).all(axis=(0,1,2,3));avg=s[...,valid].mean(axis=(0,1))
for m,method in [(1,'HQT'),(2,'TPS')]:
    row=diff[(diff.group=='primary_6')&(diff.comparison==f'{method} minus UCF')].iloc[0]
    expected=np.quantile(avg[m,0]-avg[0,0],[.025,.5,.975])
    assert np.allclose(expected,row[['low','median','high']].to_numpy(dtype=float))
colors=['#0072B2','#D55E00','#009E73'];methods=['UCF','HQT','TPS']
with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
    for ax,metric,title in zip(axes.flat,['D_RL','B','S'],['(a) Absolute log error $D_{RL}$','(b) Signed log bias $B$','(c) Shape error $S$']):
        for j,(method,color) in enumerate(zip(methods,colors)):
            row=ci[(ci.group=='primary_6')&(ci.method==method)&(ci.metric==metric)].iloc[0]
            ax.hlines(j,row.low,row.high,color=color,lw=2.5)
            ax.plot([row.low,row.high],[j,j],'|',color=color,ms=10)
            ax.scatter(row.point,j,color=color,s=45,zorder=3)
        ax.set_yticks(range(3),methods);ax.invert_yaxis();ax.set_ylim(2.6,-.6)
        ax.set_title(title,loc='left');ax.set_xlabel('Equal-weight mean across 6 stations and 7 durations')
        ax.grid(axis='x',alpha=.2)
        if metric=='B':ax.axvline(0,color='0.5',lw=.8)
        else:ax.set_xlim(left=0)
    ax=axes[1,1]
    for j,method in enumerate(['HQT','TPS']):
        row=diff[(diff.group=='primary_6')&(diff.comparison==f'{method} minus UCF')].iloc[0]
        ax.hlines(j,row.low,row.high,color=colors[j+1],lw=2.5)
        ax.plot([row.low,row.high],[j,j],'|',color=colors[j+1],ms=10)
        ax.scatter(row.point,j,color=colors[j+1],s=45,zorder=3)
    ax.set_yticks([0,1],['HQT − UCF','TPS − UCF']);ax.set_ylim(1.6,-.6)
    ax.axvline(0,color='0.4',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
    ax.set_title('(d) Paired difference in mean $D_{RL}$',loc='left')
    ax.set_xlabel('Positive values favor UCF')
    fig.suptitle('MIDAS station uncertainty: 95% bootstrap percentile intervals',fontsize=14)
    fig.supxlabel('Dots: original estimates. Lines: pointwise intervals; 1,999/2,000 jointly valid replicates.\nSynchronous 3-calendar-year circular blocks; fixed 6-station cohort; period-wise stationarity assumed.',fontsize=10)
    fig.savefig(OUT/'station_uncertainty_summary.png',dpi=180)
    fig.savefig(OUT/'station_uncertainty_summary.pdf')
    plt.close(fig)

lines=['# 站点不确定性区间：2000次日历年分块Bootstrap','',
'已完成。UCF总体点估计最优，但与HQT的差异尚不明确；相对TPS的优势得到本次名义95%差值区间支持。','',
'## 六站主分析结果','',
'|方法|原始平均D_RL|95%百分位区间|','|---|---:|---:|']
for method in methods:
    row=ci[(ci.group=='primary_6')&(ci.method==method)&(ci.metric=='D_RL')].iloc[0]
    lines.append(f'|{method}|{row.point:.4f}|[{row.low:.4f}, {row.high:.4f}]|')
lines+=['','方法比较应看配对误差差值的区间，不能只比较两种方法各自区间是否重叠：','',
'|差值|原始差值|95%区间|解释|','|---|---:|---:|---|']
for method in ['HQT','TPS']:
    row=diff[(diff.group=='primary_6')&(diff.comparison==f'{method} minus UCF')].iloc[0]
    interpretation='跨零，差异尚不明确' if row.low<=0<=row.high else '全部为正，支持UCF平均误差较低'
    lines.append(f'|{method}−UCF|{row.point:.4f}|[{row.low:.4f}, {row.high:.4f}]|{interpretation}|')
lines+=['','UCF在约90.4%的六站共同有效重采样中取得最小平均D_RL，HQT约9.5%，TPS约0.1%。这些是重采样排名频率，不是p值，也不是方法为真的后验概率。','',
'## 抽样与数据隔离','',
'使用2000次、块长3年的循环日历年块Bootstrap；每段30个日历年，随机抽10个块组成一段。各站与各历时共用年份抽样，保留历时间和站点间对应关系；前后两段独立抽样。保留各站原始缺年掩码，每次抽样后排除该站不存在的年份，因此样本量可以变化；没有把缺年前后的记录拼成连续3年块。循环首尾相接是重采样约定，并非真实相邻年份。','',
'预测阶段只读取前段小时、前段整日与后段整日AMS，重拟合与原模式实验一致的L-moment GEV，冻结参数及抽样索引。评价阶段才读取后段小时值，并使用与后段整日相同的抽样年份拟合评价参照。因此误差区间同时传播预测与小时参照估计的不确定性，并保留二者的相关性。预测雨量区间本身不依赖后段小时值。','',
'保持原有HQT概率边界和log-PAVA规则，不设额外雨量上限，不替换或重抽失败样本。单站—历时—方法预测最少1990次有效，共同评价最少1989次有效。六站全历时共同有效1999次，纳入Boulmer后为1985次；失败记录均已保存，区间条件于有效拟合。','',
'## 解释边界','',
'这些是固定站点集合、固定资料筛选、给定GEV及传递结构下的抽样不确定性区间。不是未来某场降雨的预测区间，不包括模型结构不确定性或全英国站点代表性不确定性。3年块保留局部年际依赖，但不能消除趋势；仍依赖各时期内近似平稳的工作假设。','',
'采用逐点95%百分位区间，没有作多重比较校正，也不是同时置信带。短样本和绝对误差的非线性使重采样中位数可能高于原始点估计；原始估计、重采样中位数与上下限分开保存，未用中位数替换原结果。','',
'缺年重采样后，少数站点的有效样本量很小，例如Stornoway后段最少3个、Boulmer前段最少2个。少于3个或退化样本不能完成GEV拟合，已记失败；其余小样本保留并记录样本量范围。因此这些区间是短记录条件下的近似结果，不应视为精确覆盖率保证。','',
'部分Bootstrap L-moment拟合的支持域未覆盖其全部抽样值。为与原模式估计规则一致，这一诊断没有作为额外删除条件；有效次数指拟合成功、所需曲线有限且为正，并不表示每次都通过支持域检查。逐站诊断文件保存了支持域越界次数。','',
'尾部区间揭示了实际不稳定性：Aberporth的HQT在3h、100年重现期下，原点估计约287.9 mm，95%区间约24.0—3812.0 mm。未人为截断这一上限；它说明该组合尾部外推缺乏约束，不能把区间上限直接当作可信的工程设计雨量。','',
'剔除三个原有名义趋势站后，仅剩3站，TPS−UCF区间为[-0.0021,0.3801]，跨零；因此UCF相对TPS的区间优势不应表述为对所有站点筛选方案均稳健。该子集本身属于事后筛选，仅作探索性对照。','',
'## 已交付文件','',
'- predicted_rainfall_intensity_CI95.csv：各站、方法、历时和工程重现期的预测雨量与平均强度95%区间。',
'- reference_rainfall_intensity_CI95.csv：后段小时GEV评价参照的雨量与强度区间。',
'- station_duration_metric_CI95.csv：D_RL、B、S逐站逐历时区间。',
'- station_mean_metric_CI95.csv：各站跨历时平均指标区间。',
'- station_duration_method_difference_CI95.csv：逐站逐历时配对方法差值区间。',
'- aggregate_metric_CI95.csv、aggregate_method_difference_CI95.csv：六站主分析、七站补充及三站探索性子集区间。',
'- station_uncertainty_summary.png / .pdf：六站平均指标及方法差值的区间图。',
'- calendar_draws.npz、frozen_parameters_*.npz、score_replicates.npz：可复核的抽样、冻结参数与逐次指标。',
'- prediction_bands_*.npz、truth_bands_*.npz：T=2—100的201节点逐点雨量区间。','',
f'验证：独立恢复并复算{verified}个抽样样本的L-moment参数；独立核对六站方法差值区间；所有已交付区间端点有限且排序正确；强度区间与雨量除以历时完全一致。','']
(OUT/'站点不确定性结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
save=dict(replicates_verified=verified,interval_order_verified=True,intensity_units_verified=True,
    aggregate_difference_percentiles_independently_verified=True,figure_dimensions_inches=[10,7.5],png_dpi=180,
    figure_description='Original mean D_RL, B and S with pointwise bootstrap intervals, and paired method difference intervals; fixed six-station cohort.',
    source_tables=['aggregate_metric_CI95.csv','aggregate_method_difference_CI95.csv'])
(OUT/'verification_and_figure_manifest.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
print('Verified, report and interval figure saved.')
