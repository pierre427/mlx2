"""Render measured Source stage spans and explicitly labelled logical data flow."""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT/'qualification/runs/ds41-source-lfu75-hot60-q1-dispatch-paired-20261025-r1'
OUT = ROOT/'output/ds41-profile-graph-20261025'


def stage(e, parents):
    cats = {p['category'] for p in parents}|{e['category']}
    if e['category'] == 'expert_read_wait': return 'Expert file-read wait'
    if e['category'] == 'expert_upload_lfu_other': return 'Expert admission / materialization'
    if e['category'] == 'weight_completion_wait': return 'Weight completion fences'
    if 'attention_other' in cats: return 'Attention + QKV / output / KV'
    if 'routing' in cats or 'routing_readback' in cats: return 'Router + top-k / ID readback'
    if 'expert_other' in cats: return 'Routed MoE math + assembly'
    if 'shared_expert_other' in cats: return 'Shared expert'
    if any(c.startswith('engram_') for c in cats): return 'Engram lookup + projections'
    if e['category'] == 'embedding': return 'Token embedding'
    if e['layer'] is None and (e['category'] in ('normalization','hyperconnections','dense_other','dense_compute','activation_quantization')
                              or any(p['name']=='output.weight' for p in parents)):
        return 'Final norm + vocabulary head'
    if e['category'] in ('normalization','hyperconnections'): return 'Residual mixing + normalization'
    return 'Model loop / unclassified work'


def extract():
    data = json.loads((UNIT/'analysis.json').read_text())
    totals = {p:collections.Counter() for p in ('prefill','decode')}
    waits = {p:collections.Counter() for p in totals}
    sub = {p:collections.Counter() for p in totals}
    hashes = {}
    forwards=[]
    for line in (UNIT/'model-phases.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['label']!='boundary':continue
        phase='prefill' if row['frontier']<100 else 'decode'
        p=UNIT/row['profile'];hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        boundary=json.loads(p.read_text());stack=[]
        for e in sorted(boundary['events'],key=lambda e:(e['t0'],-e['t1'])):
            while stack and not (stack[-1]['t0']<=e['t0'] and stack[-1]['t1']>=e['t1']):stack.pop()
            label=stage(e,stack);totals[phase][label]+=e['exclusive_seconds']
            waits[phase][label]+=e['input_barrier_seconds']
            if label=='Routed MoE math + assembly':sub[phase][e['category']]+=e['exclusive_seconds']
            stack.append(e)
        forwards.append({k:row[k] for k in ('frontier','input_rows','wall_seconds','profile')})
    b=data['requests']['boundary'];wall={'prefill':b['prefill_seconds'],'decode':b['decode_ms_per_interval']*31/1000}
    for p in totals:
        remainder=wall[p]-sum(totals[p].values())
        assert remainder>=0,(p,remainder)
        totals[p]['Outside measured model spans']=remainder
        assert abs(sum(totals[p].values())-wall[p])<1e-8
    result={'unit':str(UNIT),'source_snapshot_sha256':data['spec']['source_snapshot_sha256'],
            'checkpoint':data['spec']['checkpoint'],'normal_warm':{k:data['requests']['warm'][k] for k in ('prefill_seconds','decode_ms_per_interval')},
            'instrumented_wall_seconds':wall,'stage_seconds':totals,'input_dependency_barrier_seconds':waits,
            'routed_subcategories_seconds':sub,'boundary_files_sha256':hashes,'forwards':forwards,
            'note':'Stage times are exclusive nested host spans with evaluation barriers, not pure kernel durations. Input waits may complete earlier unwrapped producers. Decode totals cover31 intervals; graph divides by31.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'stage-data.json').write_text(json.dumps(result,indent=2)+'\n')
    return data,result


def render(data,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(20,25),facecolor='#f4f7fb')
    ink='#14243a';muted='#4b617a';blue='#1878b8';teal='#008b82';orange='#d1741c'
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def txt(x,y,s,size=12,color=ink,weight=None,ha='left',va='top',**kw):
        return ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,**kw)
    def box(x,y,w,h,title,body='',color=blue,title_size=13,body_size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.006',facecolor='white',edgecolor=color,linewidth=1.5))
        txt(x+.012,y+h-.012,title,title_size,color,'bold')
        if body:txt(x+.012,y+h-.033,body,body_size,linespacing=1.35)
    def arrow(x1,y1,x2,y2,label='',color=muted,rad=0):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,linewidth=1.5,color=color,connectionstyle=f'arc3,rad={rad}'))
        if label:txt((x1+x2)/2,(y1+y2)/2+.006,label,9,color,ha='center',va='bottom',bbox={'facecolor':'#f4f7fb','edgecolor':'none','pad':2})

    txt(.04,.976,'DS4.1 Source — time and data movement',29,weight='bold')
    txt(.04,.952,'100 input tokens · 32 generated tokens · batch 1 · 16K context cap · M5 Max / MLX 0.32.4',14,muted)
    txt(.04,.935,'Profile shown: ordered combine + paired original/Q1 projections. All 40 layers execute; dense/shared weights retain original precision.',11,muted)
    warm=d['normal_warm'];b=data['requests']['boundary']
    box(.04,.865,.44,.052,'NORMAL WARM REQUEST',f'Prefill  {warm["prefill_seconds"]:.3f} s  /  {100/warm["prefill_seconds"]:.2f} t/s      Decode  {warm["decode_ms_per_interval"]:.2f} ms/token  /  {1000/warm["decode_ms_per_interval"]:.2f} t/s',teal,13,12)
    box(.52,.865,.44,.052,'SYNCHRONIZED DIAGNOSTIC PASS',f'Prefill  {b["prefill_seconds"]:.3f} s      Decode  {b["decode_ms_per_interval"]:.2f} ms/token\nStage bars below use this pass; forced barriers inflate time.',orange,13,11)
    txt(.04,.843,'1  MEASURED STAGE BREAKDOWN',17,weight='bold')
    txt(.04,.824,'Exclusive nested spans; each column sums to its diagnostic wall time. Decode is the mean of 31 generated-token intervals.',11,muted)
    labels=['Token embedding','Residual mixing + normalization','Attention + QKV / output / KV','Engram lookup + projections','Router + top-k / ID readback','Expert file-read wait','Expert admission / materialization','Routed MoE math + assembly','Shared expert','Final norm + vocabulary head','Weight completion fences','Model loop / unclassified work','Outside measured model spans']
    colors=[muted,blue,blue,teal,blue,orange,orange,teal,blue,blue,orange,muted,'#98a8ba']
    pp=[d['stage_seconds']['prefill'].get(k,0)*1000 for k in labels]
    dec=[d['stage_seconds']['decode'].get(k,0)*1000/31 for k in labels]
    txt(.355,.804,'PREFILL — ms / 100-token prompt',12,blue,'bold')
    txt(.685,.804,'DECODE — ms / token',12,teal,'bold')
    for i,(label,c) in enumerate(zip(labels,colors)):
        y=.780-i*.0183
        txt(.045,y+.004,label,11,va='center')
        for x,val,mxv in ((.355,pp[i],max(pp)),(.685,dec[i],max(dec))):
            ax.add_patch(FancyBboxPatch((x,y-.003),max(.001,val/mxv*.200),.008,boxstyle='round,pad=0',facecolor=c,edgecolor='none'))
            txt(x+.207,y+.004,f'{val:,.2f}',11,c,'bold',va='center')
    txt(.04,.532,'Timing limits: materialization includes LFU/bookkeeping and evaluation; it is not an isolated copy timer. Unwrapped paired kernels,',10,muted)
    txt(.04,.519,'SwiGLU and assembly may finish at the next QAT input barrier. “Outside spans” includes surrounding runtime work and measurement gaps.',10,muted)

    txt(.04,.493,'2  DATA FLOW THROUGH A TRANSFORMER LAYER',17,weight='bold')
    txt(.04,.475,'Shared physical DRAM (unified memory). Arrows describe buffers and dependencies, not a separate PCIe transfer.',11,muted)
    box(.04,.387,.255,.068,'Attention / CSA2 / KV','QKV projections → local + pooled attention\nKV concat / owned copies; selected-row gather\nOutput projection → residual / norm',blue)
    box(.365,.387,.255,.068,'Router + expert row map','Top-6 expert IDs + scores\nIDs read back to CPU; stable expert sort\nQAT once per token row, then gather',blue)
    box(.690,.387,.270,.068,'Original / Q1 expert objects','59.993 GB original + 14.976 GB Q1\n75 GB payload + staging ceiling\nNo full dense expansion of expert weights',orange)
    arrow(.295,.422,.357,.422,'x: 5,120 F32')
    arrow(.492,.382,.492,.355,'6 routed slots / token')
    arrow(.725,.382,.60,.354,'gate/up weights',rad=.12)
    arrow(.968,.410,.968,.212,color=orange,rad=-.12)
    txt(.987,.319,'down weights',9,orange,ha='center',va='center',rotation=90)
    box(.04,.277,.255,.070,'Input row gather / QAT','Logical gather output across 40 layers:\nPP 491.520 MB  |  decode 4.915 MB\nQAT uses 32-element activation groups',teal)
    box(.365,.277,.255,.070,'Gate + up projections','Original MXFP4 or affine Q1 weights\nF32 result → BF16 round → F32\nPP 442.368 MB raw paired outputs',teal)
    box(.690,.277,.270,.070,'SwiGLU × routing score','BF16 round → F32 → expert-local FP8 QAT\nMiddle: PP 221.184 MB | decode 2.212 MB\nRounding barriers remain in place',teal)
    arrow(.492,.355,.167,.355,'row map + quantized activations',rad=.04)
    arrow(.167,.355,.167,.352)
    arrow(.295,.313,.357,.313,'gathered x')
    arrow(.620,.313,.682,.313,'gate / up')
    arrow(.825,.274,.825,.251)
    box(.04,.175,.255,.069,'Shared expert + residual','Shared expert uses original precision\nRouted + shared output merged\nHyperconnections → next layer',blue)
    box(.365,.175,.255,.069,'Ordered combine','PP99: concatenate expert output rows\nPP 486.605 MB logical concat payload\nM1: direct buffers; no concat / restore',teal)
    box(.690,.175,.270,.069,'Down projection','Middle QAT → down → BF16 round → F32\nRouted outputs: PP 491.520 MB\nDecode 4.915 MB, across 40 layers',teal)
    arrow(.682,.210,.628,.210,'routed outputs')
    arrow(.357,.210,.303,.210,'sum in expert-ID order')
    ax.plot([.033,.021,.021,.033],[.410,.410,.210,.210],color=blue,linewidth=1.2,linestyle='--')
    arrow(.021,.210,.033,.210,color=blue)
    txt(.010,.310,'same normalized x → shared branch',8,blue,ha='center',va='center',rotation=90)
    txt(.04,.158,'All activation sizes above are calculated logical F32 payloads summed over 40 layers (PP100 or one decode token). They are not measured DRAM bytes.',10,muted)
    txt(.04,.145,'Views, fusion, cache reuse and allocator behavior can change actual traffic. Gate/up/down cast intermediates and KV copies are detailed in the companion image.',10,muted)

    txt(.04,.119,'3  DISK → HOST → MLX RESIDENCY',17,weight='bold')
    diskpp=b['phases']['prefill']['expert_counters']['v41_expert_bytes_read']/1e9
    diskdec=b['phases']['decode']['expert_counters']['v41_expert_bytes_read']/31/1e6
    box(.04,.040,.275,.057,'SSD / file read counter',f'PP {diskpp:.3f} GB; decode {diskdec:.2f} MB/token\n410 Q1 misses PP; 104 over 31 decode intervals',orange,12,10)
    box(.367,.040,.275,.057,'Host file buffer + hash','Read / SHA256 → NumPy tensor views\nNo expert dequantization on the host\nAdditional header probe is outside the byte counter',orange,12,10)
    box(.695,.040,.265,.057,'MLX packed expert buffers','mx.array materialization → completion fence\nLFU retains objects; evicts dynamic Q1 objects\n18.801 MB maximum host staging reserve',orange,12,10)
    arrow(.315,.067,.359,.067)
    arrow(.642,.067,.687,.067)
    txt(.04,.021,'2026-10-25 · source snapshot 4d31882ba415… · checkpoint 927325be8884… · thermal gating waived; bounded diagnostic, not performance qualification.',9,muted)
    fig.savefig(OUT/'ds41-time-and-memory.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(OUT/'ds41-time-and-memory.svg',facecolor=fig.get_facecolor())
    plt.close(fig)


def render_movement():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    path=ROOT/'docs/audits/2026-10-25-ds41-movement/movement.json'
    d=json.loads(path.read_text())
    phases={p:{e['key']:e for e in v['entries']} for p,v in d['phases'].items()}
    groups=[('ROUTED EXPERT ACTIVATIONS',[
        ('expert_input_qat','Input FP8 QAT — physical F32 output'),
        ('routed_input_gather','Gather token rows by expert'),
        ('gate_up_projection_raw','Gate + up — raw F32 outputs'),
        ('gate_up_bf16_round_temporary','Gate + up — BF16 cast intermediates'),
        ('gate_up_rounded_f32','Gate + up — rounded F32 outputs'),
        ('weighted_middle_bf16_round_temporary','SwiGLU × score — BF16 cast'),
        ('weighted_middle_rounded_f32','SwiGLU × score — rounded F32'),
        ('middle_fp8_qat_f32','Middle FP8 QAT — physical F32 output'),
        ('down_projection_raw','Down projection — raw F32 outputs'),
        ('down_bf16_round_temporary','Down projection — BF16 cast'),
        ('down_rounded_f32','Down projection — rounded F32'),
        ('ordered_combine_concatenate','Combine — concatenate expert outputs'),
        ('ordered_combine_output','Combine — final F32 result'),
    ]),('ATTENTION, KV AND SHARED BRANCH',[
        ('attention_q_projection','Attention Q — full projection'),
        ('attention_output','Attention result, before output projections'),
        ('attention_compressed_selected_gather','Compressed KV selected-row gather (one K/V buffer)'),
        ('local_kv_band_concatenate','Local KV — concatenate prior band + new rows'),
        ('local_kv_owned_copy','Local KV — explicit retained-window copy'),
        ('compressed_pool_and_index_new_values','Pooled KV + index — new values'),
        ('shared_gate_up_raw','Shared expert — raw gate + up outputs'),
        ('shared_down_output','Shared expert — down output'),
    ]),('CONTROL AND HOST / MLX BOUNDARIES',[
        ('route_ids_device_to_host','MoE route IDs — host readback'),
        ('route_rows_and_slots_host_to_mlx','Route row + slot arrays — MLX creation'),
        ('ordered_offsets_host_to_mlx','Combine offsets — MLX creation'),
        ('csa2_selected_indices','CSA2 selected indices — device only'),
        ('engram_returned_f32_host_rows','Engram hit rows — host gather; same-size MLX creation'),
    ])]
    fig=plt.figure(figsize=(18,21),facecolor='#f4f7fb');ax=fig.add_axes([0,0,1,1]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ink='#14243a';muted='#4b617a';teal='#008b82';orange='#bc6b18'
    def text(x,y,s,size=12,color=ink,weight=None,**kw):ax.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',**kw)
    text(.04,.975,'DS4.1 — buffer movement detail',28,weight='bold')
    text(.04,.950,'Same paired-profile capture · values sum over all 40 layers · decimal MB · PP = 99-row forward + 1-row tail',12,muted)
    ax.add_patch(FancyBboxPatch((.04,.864),.92,.065,boxstyle='round,pad=.007',facecolor='white',edgecolor=orange))
    text(.055,.918,'These are logical tensor sizes, not measured DRAM traffic or peak live memory.',15,orange,'bold')
    text(.055,.895,'A gather / concat / explicit copy has a known logical output size. Casts and pointwise buffers may fuse or reuse storage.\nDo not add the table into a bandwidth claim. Copy-only latency is not isolated by the existing trace.',12,muted)
    text(.045,.841,'Operation / buffer',12,muted,'bold')
    for x,s in ((.665,'PP100 MB'),(.785,'First decode MB'),(.91,'Mean decode MB')):text(x,.841,s,11,muted,'bold',ha='right')
    y=.817
    for title,rows in groups:
        text(.045,y,title,12,teal,'bold');y-=.024
        for key,label in rows:
            cls=phases['pp100'][key]['classification'];c=orange if 'cast' in cls else ink
            ax.plot([.045,.94],[y-.018,y-.018],color='#dce4ec',linewidth=.6)
            text(.05,y,label,11,c)
            for x,p,div in ((.665,'pp100',1),(.785,'decode1_first',1),(.91,'decode31',31)):
                text(x,y,f"{phases[p][key]['bytes']/div/1e6:,.4f}",11,c,ha='right')
            y-=.022
        y-=.014
    # Compact explanatory cards below the derived table.
    ax.add_patch(FancyBboxPatch((.04,.070),.44,.063,boxstyle='round,pad=.006',facecolor='white',edgecolor=teal))
    text(.055,.122,'KV capacity ≠ KV movement',13,teal,'bold')
    text(.055,.103,'Retained KV: 10.813 MB after PP → 13.120 MB after decode.\n31 decode steps: 588.186 MB of logical local-KV concat + copies.\nPools are physically F32 despite quantized values.',10,muted)
    ax.add_patch(FancyBboxPatch((.52,.070),.44,.063,boxstyle='round,pad=.006',facecolor='white',edgecolor=orange))
    text(.535,.122,'Expert-file ownership transition',13,orange,'bold')
    text(.535,.103,'PP: 2.721 GB reported file reads → 2.721 GB packed MLX payload.\nCPU hashes full files; NumPy tensors are views. Q1 header probes\nadd 370,640 requested bytes outside the PP file-read counter.',10,muted)
    text(.04,.050,'Interpretation: low-byte host readbacks can still synchronize the GPU. Direct one-row combine avoids the prefill concatenate.\nThe compressed-attention gather is reused for K and V; it is counted once. Decode KV/pool sizes grow, so first and mean differ.',11,muted)
    text(.04,.018,'Data: movement.json + frozen source 4d31882ba415… · exact formulas and exclusions retained with the images · 2026-10-25',9,muted)
    fig.savefig(OUT/'ds41-memory-detail.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(OUT/'ds41-memory-detail.svg',facecolor=fig.get_facecolor());plt.close(fig)
    (OUT/'movement-data.json').write_bytes(path.read_bytes())


if __name__=='__main__':
    data,result=extract();render(data,result);render_movement()
    print(json.dumps({'output':str(OUT),'stage_ms':{p:{k:round(v*1000/(31 if p=='decode' else 1),3) for k,v in r.items()} for p,r in result['stage_seconds'].items()}},indent=2))
