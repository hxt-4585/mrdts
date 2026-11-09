"""Snapshot active verification logs without touching simulation state."""
import argparse
from datetime import datetime, timezone, timedelta
from itertools import islice
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_average(run, snapshots):
    fig, axes = plt.subplots(1,3,figsize=(15,4.9))
    colors=['#3476A8','#C77536','#27877C']
    metadata=dict(definition='Per-slot mean over all 12 UAV queues; cumulative mean over all elapsed slots and UAVs.',
                  interpretation='Intermediate snapshot; cumulative mean includes startup backlog.',snapshots=[])
    arrays={}
    for index,(progress,q) in enumerate(snapshots):
        x=np.arange(1,len(q)+1)
        mean=q.mean(1)
        cumulative=np.cumsum(mean)/x
        label=f'种子 {progress["epoch"]}'
        style=dict(color=colors[index],label=label,linewidth=1.1,
                   linestyle='-' if index==0 else '--')
        axes[0].plot(x,mean,**style)
        axes[1].plot(x,cumulative,**style)
        axes[2].plot(x[-1000:],mean[-1000:],**style)
        metadata['snapshots'].append(dict(epoch=progress['epoch'],slots=len(q),
                     instantaneous_mean_queue_j=float(mean[-1]),
                     cumulative_mean_queue_j=float(cumulative[-1]),
                     recent_1000_mean_queue_j=float(mean[-1000:].mean()),
                     recent_1000_nonzero_fraction=float((mean[-1000:]>1e-6).mean())))
        arrays[f'epoch_{progress["epoch"]}_mean_j']=mean
        arrays[f'epoch_{progress["epoch"]}_cumulative_j']=cumulative
    titles=['每个时隙：12 架 UAV 的平均队列','从起点到当前：累计平均队列','最近 1000 时隙：平均队列放大']
    for ax,title in zip(axes,titles):
        ax.set(title=title,xlabel='时隙',ylabel='平均队列（J）')
        ax.grid(axis='y',color='#E2E6E9',linewidth=.65)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.set_ylim(bottom=0.)
        ax.margins(x=.015)
    axes[2].ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('平均能量虚拟队列｜联合训练模型 · 预算 128.45 J · V = 100000',fontsize=15,y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
    counts=' / '.join(f'{len(q):,}' for _,q in snapshots)
    fig.text(.5,.025,f'快照：{counts} 时隙。平均包含全部 12 架 UAV（含零队列）；累计平均含启动积压，不能单独作为稳态证据。',
             ha='center',fontsize=9,color='#4E5964')
    fig.subplots_adjust(left=.06,right=.98,bottom=.17,top=.76,wspace=.30)
    output=run/'figures'
    output.mkdir(exist_ok=True)
    stem='average_queue_snapshot_'+'_'.join(str(len(q)) for _,q in snapshots)
    for ext in ['png','svg','pdf']:
        fig.savefig(output/f'{stem}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    np.savez_compressed(output/f'{stem}_data.npz',**arrays)
    (output/f'{stem}.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(image=str((output/f'{stem}.png').resolve()),metadata=metadata),ensure_ascii=False))


def plot(run, average=False):
    snapshots = []
    for path in sorted((run/'confirm'/'joint').glob('*/progress.json')):
        progress = json.loads(path.read_text(encoding='utf-8'))
        n = progress['slots_completed']
        with (path.parent/'slots.jsonl').open(encoding='utf-8') as stream:
            rows = list(map(json.loads, islice(stream, n)))
        q = np.array([row['queue_j'] for row in rows])
        snapshots.append((progress, q))
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.unicode_minus':False, 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'axes.linewidth':.7})
    if average:
        plot_average(run,snapshots)
        return
    fig, axes = plt.subplots(2, len(snapshots), figsize=(12.8,7.4), squeeze=False)
    colors = {1:'#3476A8', 2:'#C77536', 9:'#27877C'}
    maximum = max(q.max() for _,q in snapshots)
    detail_max = max(q[-1000:, [1,9]].max() for _,q in snapshots)
    metadata = dict(captured_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    backend='Python / matplotlib', archetype='quantitative grid',
                    conclusion='Startup backlog is declining; small nonzero fluctuations need a separate scale.',
                    definition='Actual per-UAV energy virtual queues after slot budget deduction, UAV IDs are zero-based.',
                    smoothing='None; every recorded slot is plotted.',
                    interpretation='Intermediate verification snapshot, not a final stability conclusion.', snapshots=[])
    arrays = {}
    for col, (progress, q) in enumerate(snapshots):
        n = len(q)
        x = np.arange(1,n+1)
        full, detail = axes[:,col]
        for member in range(q.shape[1]):
            if member not in colors:
                full.plot(x,q[:,member],color='#A6ADB4',linewidth=.65,alpha=.65)
        for member in [1,9,2]:
            full.plot(x,q[:,member],color=colors[member],linewidth=1.05,zorder=3)
        full.set(title=f'任务种子 {progress["epoch"]} · 已完成 {n:,} 时隙',
                 ylabel='能量虚拟队列 Q（J）',xlabel='时隙',xlim=(1,n),ylim=(-maximum*.015,maximum*1.08))
        full.annotate(f'UAV 2 当前：{q[-1,2]:.2f} J',xy=(n,q[-1,2]),
                      xytext=(.52,.74),textcoords='axes fraction',fontsize=10,color=colors[2],
                      arrowprops=dict(arrowstyle='-',color=colors[2],linewidth=.8))
        start = max(0,n-1000)
        for member in [1,9]:
            detail.plot(x[start:],q[start:,member],color=colors[member],linewidth=.85,alpha=.9)
        detail.set(title='最近 1000 时隙：UAV 1、9 的小幅队列',ylabel='能量虚拟队列 Q（J）',
                   xlabel='时隙',xlim=(x[start],n),ylim=(-.004,detail_max*1.15))
        for ax in [full,detail]:
            ax.grid(axis='y',color='#E2E6E9',linewidth=.65,zorder=0)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        tail=q[-1000:]
        metadata['snapshots'].append(dict(epoch=progress['epoch'], slots=n,
                 final_queue_j=q[-1].tolist(), recent_1000_mean_j=tail.mean(0).tolist(),
                 recent_1000_nonzero_fraction=(tail>1e-6).mean(0).tolist()))
        arrays[f'epoch_{progress["epoch"]}_queues_j']=q
    handles=[Line2D([0],[0],color=colors[i],lw=1.6,label=f'UAV {i}') for i in [1,2,9]]
    handles.append(Line2D([0],[0],color='#A6ADB4',lw=1.2,label='其余 9 架 UAV'))
    fig.suptitle('当前队列走势｜联合训练模型的预算迁移验证',fontsize=16,y=.98)
    fig.text(.5,.926,'每时隙预算 128.45 J · V = 100000 · 每条轨迹途中不重置队列 · UAV ID 从 0 开始',
             ha='center',fontsize=10,color='#4E5964')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.90),ncol=4,frameon=False)
    fig.text(.5,.018,'上排：12 架 UAV 的原始逐时隙队列；下排使用独立纵轴放大小幅波动。当前为中间快照，非最终稳定性结论。',
             ha='center',fontsize=9,color='#4E5964')
    fig.subplots_adjust(left=.075,right=.975,bottom=.105,top=.81,wspace=.25,hspace=.49)
    output=run/'figures'
    output.mkdir(exist_ok=True)
    stem='queue_snapshot_'+ '_'.join(str(len(q)) for _,q in snapshots)
    for ext in ['png','svg','pdf']:
        fig.savefig(output/f'{stem}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    np.savez_compressed(output/f'{stem}_data.npz',**arrays)
    (output/f'{stem}.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(image=str((output/f'{stem}.png').resolve()),metadata=metadata),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--average',action='store_true')
    args=parser.parse_args()
    plot(args.run,args.average)
