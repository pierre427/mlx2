# SPDX-License-Identifier: MIT
"""Original V4.1 embedded DSpark equations and complete3x5 Markov paths.

The three stages share target embedding/head but own their original packed
E128/top3 experts. Target features are attention-input HC means, rounded to
the checkpoint BF16 boundary. This is not the older Vision-Exp DSpark law.
Provenance: deepseek-v41-dspark-ast.json. No runtime selection on import.
"""
import hashlib
import math
from copy import copy
from pathlib import Path

from ..drafters.deepseek_v4_dspark import DSparkHistory, markov_chain


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def ranked_markov_paths(base_logits,hidden,seed,*,lookup,project,confidence,branches=3,forbidden_root=(),forbidden_at_depth=None):
    """One trained block, distinct ranked roots, full conditional continuations.

    Confidence is telemetry only and never truncates the requested five rows
    or substitutes for target verification. Each branch consumes its own
    previous-token embedding, including at depths2..5.
    """
    import mlx.core as mx
    if base_logits.ndim!=2 or base_logits.shape[0]!=5 or hidden.ndim!=2 or hidden.shape[0]!=5:
        raise ValueError('V4.1 DSpark requires exactly five aligned draft rows')
    if branches!=3 or type(seed)is not int or not 0<=seed<base_logits.shape[-1]:
        raise ValueError('V4.1 AST DSpark requires three roots and a valid anchor')
    def step(previous,row):
        embed=lookup(previous).astype(mx.float32).reshape(-1)
        logits=base_logits[row].astype(mx.float32)+project(embed[None]).reshape(-1)
        score=confidence(mx.concatenate((hidden[row].astype(mx.float32),embed))[None]).reshape(-1)[0]
        mx.eval(logits,score)
        if not bool(mx.all(mx.isfinite(logits)).item())or not math.isfinite(float(score.item())):
            raise ValueError('Nonfinite embedded DSpark Markov/confidence output')
        return logits,float(score.item())
    root,root_confidence=step(seed,0)
    extra=()if forbidden_at_depth is None else forbidden_at_depth(0)
    forbidden=tuple(sorted({int(x)for x in (*forbidden_root,*extra)}))
    if any(not 0<=x<root.shape[0]for x in forbidden):raise ValueError('Invalid forbidden root token')
    if root.shape[0]-len(forbidden)<3:raise ValueError('Not enough legal DSpark root alternatives')
    if forbidden:
        root=mx.put_along_axis(root,mx.array(forbidden,mx.int32),mx.full((len(forbidden),),-mx.inf),axis=0)
    # Stable descending logits retain ascending token-ID ties.
    roots=mx.argsort(-root)[:3].tolist()
    paths=[];confidences=[]
    for selected in roots:
        tokens=[int(selected)];scores=[root_confidence]
        for row in range(1,5):
            logits,score=step(tokens[-1],row)
            masked=()if forbidden_at_depth is None else tuple(sorted(set(forbidden_at_depth(row))))
            if masked:
                if any(type(t)is not int or not 0<=t<logits.shape[0]for t in masked)or len(masked)>=logits.shape[0]:
                    raise ValueError('No valid conditional DSpark continuation')
                logits=mx.put_along_axis(logits,mx.array(masked,mx.int32),mx.full((len(masked),),-mx.inf),axis=0)
            tokens.append(int(mx.argmax(logits).item()));scores.append(score)
        paths.append(tuple(tokens));confidences.append(tuple(scores))
    return tuple(paths),tuple(confidences)


class SourceDSparkBackbone:
    """Bound loaded head, with committed per-request support owned separately."""
    qualified=False
    supports_sampling=False
    supports_ordered_alternatives=True

    def __init__(self,target,weights,record,*,check_live):
        if not callable(check_live)or check_live()is not True:raise ValueError('DSpark needs a live owner')
        from ..models.deepseek_v41 import Attention, Block
        self.target,self.weights,self.record,self.check_live=target,weights,record,check_live
        self.config=record['config']
        self.identity=record['fingerprint']+':target='+target.identity+':implementation='+implementation_identity()
        self.model=copy(target)
        self.model.args=record['artifact']['args'];self.model.weights=weights
        self.model.hc_policy='reference';self.model.hc_coeff_policy='reference'
        self.model.source_hc_boundary_policy='reference'
        self.model.pointwise_policy='reference';self.model.hc_epilogue_policy='reference'
        self.model.attention_epilogue_policy='reference'
        self.model.qat_policy='reference'
        self.model.scheduled_hc_calls={};self.model.scheduled_kv_calls={}
        # Construct actual V4.1 HC/mixes/router/sharedexpert mechanisms; the
        # attention replacement below supplies only the noncausal draft KV law.
        self.blocks=[Block(self.model,i)for i in range(3)]
        class DraftAttention(Attention):
            def target_kv(self,main,positions):
                m=self.model
                kv=m.norm(self.linear(main,'attn_kv'),self.kv_norm)[:,None]
                return self.rotate_quantized(kv,'fp8',positions=positions)[:,0]

            def __call__(self,x,state,cache,shared):
                import mlx.core as mx

                from ..models.deepseek_v4 import attend
                m,c=self.model,self.c
                length=x.shape[1];position=shared['dspark_position']
                positions=tuple(range(position,position+length))
                qr=m.norm(self.linear(x,'attn_q_a'),self.q_norm)
                q=self.linear(qr,'attn_q_b').reshape(1,length,c.num_attention_heads,c.head_dim).transpose(0,2,1,3)
                # V4.1 intentionally has no per-head query RMSNorm.
                q=self.rotate(q,positions=positions)
                current=self.target_kv(x,positions)
                support=shared['dspark_support_kv']
                kv=current if support is None else mx.concatenate((support,current),axis=1)
                mask=mx.ones((1,1,length,kv.shape[1]),mx.bool_)
                heads=m.round(attend(q,kv[:,None],mask,sinks=self.sinks))
                heads=self.rotate(heads,positions=positions,inverse=True)
                grouped=m.qat(heads.transpose(0,2,1,3).reshape(1,length,c.o_groups,-1),'fp8')
                low=m.round(self.w.grouped(grouped,self.prefix+'attn_output_a.weight',groups=c.o_groups))
                return self.linear(low.reshape(1,length,-1),'attn_output_b')
        for i,block in enumerate(self.blocks):block.attn=DraftAttention(self.model,i)
        self.main_norm=weights.array('main_norm.weight');self.final_norm=weights.array('draft_norm.weight')
        self.stats={'feature_rows': 0,'backbone_blocks': 0,'markov_paths': 0,'markov_steps': 0}
        self._closed=False

    def make_history(self):
        from .deepseek_v4_dspark import DSparkHistory
        return DSparkHistory(self.identity,3,self.config.window)

    def _live(self):
        if self._closed or self.check_live()is not True:raise RuntimeError('DSpark execution owner lost or closed')

    def record_features(self,hidden,position,history):
        """Append only committed target attention-input captures37/38/39."""
        import mlx.core as mx
        self._live();c=self.config
        if history.identity!=self.identity or hidden.ndim!=3 or hidden.shape[0]!=1 or hidden.shape[-1]!=3*c.hidden_size:
            raise ValueError('V4.1 DSpark feature/history binding differs')
        if type(position)is not int or position!=history.next_position:raise ValueError('DSpark support frontier differs')
        if not hidden.shape[1]:return
        skip=max(0,hidden.shape[1]-c.window)
        # Authored captured HC means are BF16, despite runtime target taps
        # being transported in FP32 containers.
        selected=self.model.round(hidden[:,skip:])
        main=self.model.norm(self.weights.linear('main_proj.weight')(selected),self.main_norm)
        positions=tuple(range(position+skip,position+hidden.shape[1]))
        kv=tuple(block.attn.target_kv(main,positions)for block in self.blocks)
        mx.eval(main,kv)
        history.append(main,kv,position,skipped_prefix=skip)
        self.stats['feature_rows']+=hidden.shape[1]

    def propose_paths(self,anchor,history,*,forbidden_root=(),forbidden_at_depth=None):
        import mlx.core as mx
        self._live();c=self.config;m=self.model
        if history.identity!=self.identity or history.next_position<1 or history.main_history is None:
            raise ValueError('DSpark requires committed target context')
        if type(anchor)is not int or not 0<=anchor<c.vocab_size:raise ValueError('Invalid DSpark anchor')
        ids=mx.array([[anchor]+[c.noise_token_id]*4],mx.uint32)
        x=m.round(self.target.weights.embedding(ids))
        h=mx.broadcast_to(x[:,:,None],(1,5,m.args.hc_mult,c.hidden_size))
        pre=mx.broadcast_to(mx.array([1.,0.,0.,0.],mx.float32),(1,5,4))
        with self.weights.execution_scope_for_rows(5):
            for stage,block in enumerate(self.blocks):
                self._live()
                with self.weights.dense_window():
                    shared={'dspark_position': history.next_position,'dspark_support_kv': history.stage_kv[stage]}
                    h,pre=block(h,pre,None,None,shared)
                    self.weights.evaluate_dense_window(h,pre)
                self.stats['backbone_blocks']+=1
            hidden=m.collapse(h,pre)
            normalized=m.norm(hidden,self.final_norm)
            logits=self.target.weights.linear('output.weight')(normalized).reshape(5,c.vocab_size)
            mx.eval(logits,hidden)
        paths,confidence=ranked_markov_paths(logits,hidden.reshape(5,c.hidden_size),anchor,
            lookup=lambda token:self.weights.array('markov_embed.weight')[token],
            project=self.weights.linear('markov_head.weight'),confidence=self.weights.linear('confidence.weight'),
            forbidden_root=forbidden_root,forbidden_at_depth=forbidden_at_depth)
        self.stats['markov_paths']+=3;self.stats['markov_steps']+=15
        return paths,confidence

    def receipt(self):
        return {'schema': 'mlx2.v41-source-dspark.v1','head_fingerprint': self.record['fingerprint'],
            'source_checkpoint': self.record['checkpoint'],'target_identity': self.target.identity,
            'target_taps': list(self.config.target_layers),'feature_law': 'attention_input_hc_mean_bf16',
            'draft_stages': 3,'trained_rows': 5,'paths': 3,'depth': 5,'expert_count': 128,'top_k': 3,
            'expert_format': 'original_MXFP4_E2M1_E8M0_group32','target_hotmap_applies': False,
            'shared_target_embedding_and_head': True,'stats': dict(self.stats),'qualified': False,
            'admission': getattr(self,'admission',None)}

    def close(self):
        if self._closed:return
        self._live();self.weights.close();self._closed=True
        restore=getattr(self,'_target_headroom_restore',None)
        if restore is not None:
            stream,previous,installed,old_protected,protected=restore
            if installed is not None and stream._adaptive_headroom==installed:
                stream._adaptive_headroom=previous
            if old_protected is not None and stream._adaptive_protected_bytes==protected:
                stream._adaptive_protected_bytes=old_protected
            self._target_headroom_restore=None


# Serial embedded B1 proposals preserve their independent admitted owner.
class DeepSeekV41DSparkBackbone:
    qualified = False

    def __init__(self, config, target, weights, identity):
        from ...adapters.deepseek_v41_dspark_artifact import draft_config
        from ..models.deepseek_v41 import DeepSeekV41Model

        self.config, self.target, self.w, self.identity = (
            config,
            target,
            weights,
            identity,
        )
        self.target_weights, self.target_config = target.weights, target.args
        self.math = DeepSeekV41Model(
            draft_config(target.args), weights, identity=identity
        )
        self.history = DSparkHistory(identity, config.stages, 128)

    def record_features(self, captures, position, *, history=None):
        import mlx.core as mx

        verify = getattr(self.w, "verify_source", None)
        if verify is not None:
            verify()
        if set(captures) != set(self.config.target_layers):
            raise ValueError("V4.1 DSpark requires pre-attention taps37/38/39")
        values = [captures[i] for i in self.config.target_layers]
        shape = values[0].shape
        if (
            len(shape) != 3
            or shape[0] != 1
            or shape[2] != self.target.args.hidden_size
            or any(v.shape != shape for v in values)
        ):
            raise ValueError("DSpark capture geometry mismatch")
        skip = max(0, shape[1] - 128)
        m = self.math
        main = m.norm(
            m.linear(
                m.round(mx.concatenate([v[:, skip:] for v in values], -1)),
                "main_proj.weight",
            ),
            self.w.array("main_norm.weight"),
        )
        positions = tuple(range(position + skip, position + skip + main.shape[1]))
        kv = tuple(self._target_kv(block, main, positions) for block in m.layers)
        h = self.history if history is None else history
        if h.identity != self.identity:
            raise ValueError("DSpark history binding mismatch")
        h.append(main, kv, position, skipped_prefix=skip)

    def _target_kv(self, block, main, positions):
        m = self.math
        kv = m.norm(m.linear(main, block.prefix + "attn_kv.weight"), block.attn.kv_norm)
        return block.attn.rotate_quantized(kv[:, None], "fp8", positions=positions)[
            :, 0
        ]

    def _stage(self, block, h, pre, support, position):
        import mlx.core as mx

        from ..models.deepseek_v4 import attend

        m, c = self.math, self.math.args
        count = h.shape[1]
        next_pre, post, comb = block.mixes(h, "attn")
        x = m.norm(m.collapse(h, pre), block.attn_norm)
        qr = m.norm(m.linear(x, block.prefix + "attn_q_a.weight"), block.attn.q_norm)
        q = (
            m.linear(qr, block.prefix + "attn_q_b.weight")
            .reshape(1, count, c.num_attention_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        positions = tuple(range(position, position + count))
        q = block.attn.rotate(q, positions=positions)
        draft = self._target_kv(block, x, positions)
        kv = mx.concatenate((support, draft), 1)
        out = attend(
            q,
            kv[:, None],
            mx.ones((1, 1, count, kv.shape[1]), mx.bool_),
            sinks=block.attn.sinks,
        )
        out = block.attn.rotate(m.round(out), positions=positions, inverse=True)
        grouped = m.qat(
            out.transpose(0, 2, 1, 3).reshape(1, count, c.o_groups, -1), "fp8"
        )
        low = m.round(
            self.w.grouped(
                grouped, block.prefix + "attn_output_a.weight", groups=c.o_groups
            )
        )
        branch = m.linear(
            low.reshape(1, count, -1), block.prefix + "attn_output_b.weight"
        )
        h = m.expand(branch, h, post, comb)
        ffn_pre, post, comb = block.mixes(h, "ffn")
        x = m.norm(m.collapse(h, next_pre), block.ffn_norm)
        ids, scores = block.router(x)
        routed = self.w.moe(
            x.reshape(-1, c.hidden_size),
            ids.reshape(-1, 3),
            scores.reshape(-1, 3),
            block.prefix[:-1],
        ).reshape(x.shape)
        return block.finish(h, routed, block.shared_expert(x), post, comb), ffn_pre

    def propose(
        self,
        seed,
        *,
        position,
        history=None,
        max_proposals=None,
        confidence_threshold=0.0,
    ):
        import mlx.core as mx

        verify = getattr(self.w, "verify_source", None)
        if verify is not None:
            verify()
        c = self.config
        h = self.history if history is None else history
        limit = c.block_size if max_proposals is None else max_proposals
        if (
            h.identity != self.identity
            or h.next_position != position
            or h.main_history is None
            or type(seed) is not int
            or not 0 <= seed < self.target.args.vocab_size
            or type(limit) is not int
            or not 1 <= limit <= 5
        ):
            raise ValueError("DSpark proposal binding/frontier/count mismatch")
        tokens = mx.array([[seed] + [c.noise_token_id] * 4], mx.uint32)
        m = self.math
        embedded = m.round(self.w.embedding(tokens))
        value = mx.broadcast_to(
            embedded[:, :, None], (1, 5, m.args.hc_mult, m.args.hidden_size)
        )
        pre = mx.broadcast_to(mx.array([1.0, 0.0, 0.0, 0.0]), value.shape[:-1])
        for i, block in enumerate(m.layers):
            value, pre = self._stage(block, value, pre, h.stage_kv[i], position)
        hidden = m.collapse(value, pre)[:, :limit]
        logits = self.target.weights.linear("output.weight")(
            m.norm(hidden, self.w.array("output_norm.weight"))
        ).reshape(limit, -1)
        return markov_chain(
            logits,
            hidden.reshape(limit, -1),
            seed,
            lookup_w1=lambda token: self.w.embedding(
                mx.array([token], mx.uint32), "markov_embed.weight"
            )[0],
            project_w2=self.w.linear("markov_head.weight"),
            confidence_project=self.w.linear("confidence.weight"),
            threshold=confidence_threshold,
        )

    def close(self):
        self.w.close()
