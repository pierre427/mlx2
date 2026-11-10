"""V4.1 resident packed weights; disk-only Engram tables cannot be uploaded."""

import mmap
import math
import os
from contextlib import nullcontext

from ..runtime.gguf.reader import GGUFError, GGUFFile
from ..runtime.gguf.weights import GGUFWeights


class ResidentFile(GGUFFile):
    def __init__(self, path):
        super().__init__(path)
        self.resident_end = max(
            self.data_offset + t.offset + t.nbytes
            for t in self.tensors.values()
            if ".engram_embd." not in t.name
        )
        if any(
            self.data_offset + t.offset < self.resident_end
            for t in self.tensors.values()
            if ".engram_embd." in t.name
        ):
            self.close()
            raise GGUFError("V4.1 Engram tables overlap resident extent")

    def packed_view(self, name):
        if ".engram_embd." in name:
            raise GGUFError(
                "Engram table is disk-only; use bounded positional row reads"
            )
        tensor = self.tensors[name]
        if self._mapping is None:
            self._mapping = mmap.mmap(
                self._file.fileno(), self.resident_end, access=mmap.ACCESS_READ
            )
        start = self.data_offset + tensor.offset
        return memoryview(self._mapping)[start : start + tensor.nbytes]


class DeepSeekV41Weights(GGUFWeights):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.file.close()
        self.file = ResidentFile(path)
        self.metadata, self.tensors = self.file.metadata, self.file.tensors
        self.expert_stream = None
        self.dense_pages = None
        self._parked = False
        self._resident_resume_bytes = 0
        from ..runtime.immutable_weight_pages import file_stamp
        self._source_stat = file_stamp(os.fstat(self.file._file.fileno()))
        try:
            self.verify_source()
        except BaseException:
            self.file.close()
            raise

    @staticmethod
    def _pageable(tensor):
        return (len(tensor.shape) >= 2 and tensor.ggml_type != 19
                and not any(part in tensor.name for part in (".engram_embd.", "hc_", "norm", ".bias", "attn_sinks")))

    def dense_paging_plan(self, cache_bytes):
        """CPU-only upper bounds before model allocation; tensor pages are exact."""
        from ..runtime.immutable_weight_pages import ImmutablePage

        pages, stages = {}, {}
        anchor = 0
        for name,t in self.tensors.items():
            if ".engram_embd." in name or t.ggml_type == 19:
                continue
            if not self._pageable(t):
                # Small HC/norm anchors may retain a native source plus FP32
                # prepared values. This is a reservation, not measured memory.
                anchor += t.nbytes + (4*math.prod(t.shape) if t.ggml_type != 0 else 0)
                continue
            extra = (4*math.prod(t.shape) if t.ggml_type in (1,30)
                     and len(t.shape) == 2 and name != "token_embd.weight" else 0)
            page = ImmutablePage(self.file.data_offset+t.offset,t.nbytes,t.nbytes+extra,
                                 f"gguf:{t.ggml_type}:{','.join(map(str,t.shape))}")
            pages[name] = page
            stage = '.'.join(name.split('.')[:2]) if name.startswith("blk.") else name
            stages[stage] = stages.get(stage,0)+page.resident_bound_bytes
        if not pages:
            raise GGUFError("no dense pages declared")
        minimum = max(stages.values())
        if type(cache_bytes) is not int or cache_bytes < minimum:
            raise GGUFError(f"dense page cache requires at least {minimum} bytes for one stage")
        staging = max(p.source_bytes for p in pages.values())+(16 << 20)
        return dict(pages=pages,page_count=len(pages),source_bytes=sum(p.source_bytes for p in pages.values()),
                    cache_bytes=cache_bytes,minimum_cache_bytes=minimum,staging_bound_bytes=staging,
                    resident_anchor_bound_bytes=anchor+(1 << 20),
                    resident_remainder_bound_bytes=anchor+(1 << 20)+cache_bytes+staging)

    def enable_dense_paging(self, cache_bytes, *, binding, synchronize=None):
        import mlx.core as mx
        from ..runtime.immutable_weight_pages import ImmutableWeightPages

        if self.dense_pages is not None or self._parked or self.expert_stream is None:
            raise GGUFError("dense paging requires a fresh streamed-expert weight owner")
        plan = self.dense_paging_plan(cache_bytes)
        if any(name in plan["pages"] for name in (*self._dense,*self._packed)):
            raise GGUFError("dense paging must be bound before resident matrix upload")
        self.dense_plan = {key:value for key,value in plan.items() if key != "pages"}
        self.dense_pages = ImmutableWeightPages(self.file.path,plan["pages"],binding=binding,
            expected_stat=self._expected_stat,cache_bytes=cache_bytes,
            materialize=self._materialize_page,evict=self._evict_page,
            synchronize=mx.synchronize if synchronize is None else synchronize,
            minimum_bytes=plan["minimum_cache_bytes"])
        return self.dense_receipt()

    def _materialize_page(self, name, raw):
        import mlx.core as mx
        import numpy as np
        from ..runtime.gguf.packed_metal import PackedMatrix

        t = self.tensors[name]
        if t.ggml_type in (0,1,24,25,26,27,30):
            dtype={0:"<f4",1:"<f2",24:"i1",25:"<i2",26:"<i4",27:"<i8",30:"<u2"}[t.ggml_type]
            value=mx.array(raw.view(np.dtype(dtype)).reshape(t.numpy_shape))
            if t.ggml_type == 30:value=value.view(mx.bfloat16)
            mx.eval(value)
            self._dense[name]=value
        else:
            value=PackedMatrix(mx.array(raw),t)
            mx.eval(value.data)
            self._packed[name]=value
        return value

    def _evict_page(self, name):
        self._packed.pop(name,None)
        self._dense.pop(name,None)
        self._native_linear_fp32.pop(name,None)
        for key in tuple(self._grouped):
            if key[0] == name:self._grouped.pop(key)

    def dense_window(self):
        if self._parked:raise GGUFError("V4.1 weight owner is parked")
        return self.dense_pages.window() if self.dense_pages is not None else nullcontext()

    def evaluate_dense_window(self, *values):
        if self.dense_pages is not None:
            import mlx.core as mx
            def arrays(value):
                if isinstance(value,mx.array):
                    yield value
                elif isinstance(value,dict):
                    for item in value.values():yield from arrays(item)
                elif isinstance(value,(tuple,list)):
                    for item in value:yield from arrays(item)
            mx.eval(list(arrays(values)))

    def bind_dense_synchronizer(self, synchronize):
        if self.dense_pages is not None:
            self.dense_pages.bind_owner()
            if not callable(synchronize):raise ValueError("dense synchronizer must be callable")
            self.dense_pages.synchronize=synchronize

    def verify_source(self):
        from ..runtime.immutable_weight_pages import REBOOT_VOLATILE_STAMP_KEYS, file_stamp
        if (file_stamp(os.stat(self.file.path,follow_symlinks=False)) != self._source_stat
                or file_stamp(os.fstat(self.file._file.fileno())) != self._source_stat
                or self._expected_stat is not None and (not set(self._expected_stat) - REBOOT_VOLATILE_STAMP_KEYS
                    or any(self._source_stat.get(k) != v for k,v in self._expected_stat.items() if k not in REBOOT_VOLATILE_STAMP_KEYS))):
            raise GGUFError("V4.1 weight source changed after admission")

    def resident_bytes(self):
        """Retained payload estimate, excluding model anchors and allocator slack."""
        values=list(self._dense.values())+[p.data for p in self._packed.values()]
        total=sum(v.nbytes for v in {id(v):v for v in values}.values())
        total+=sum(v.nbytes for name,v in self._native_linear_fp32.items() if self.tensors[name].ggml_type != 0)
        return total

    def resume_reservation_bytes(self):
        if self.dense_pages is not None:
            return self.dense_plan["resident_anchor_bound_bytes"]+self.dense_pages.cache_bytes+self.dense_plan["staging_bound_bytes"]
        return max(self._resident_resume_bytes,self.resident_bytes())

    @property
    def minimum_cache_bytes(self):
        return self.dense_pages.minimum_bytes if self.dense_pages is not None else 0

    @property
    def maximum_cache_bytes(self):
        return self.dense_pages.maximum_bytes if self.dense_pages is not None else 0

    @property
    def cache_budget_bytes(self):
        return self.dense_pages.cache_bytes if self.dense_pages is not None else 0

    def resize_cache_budget(self, budget):
        import mlx.core as mx
        if self.dense_pages is None:raise ValueError("dense paging is not selected")
        result=self.dense_pages.resize(budget)
        mx.clear_cache()
        return result

    def reserved_bytes(self):
        return 0 if self._parked else self.resume_reservation_bytes()

    def dense_receipt(self):
        return dict(mode="tensor_pages" if self.dense_pages is not None else "resident",parked=self._parked,
                    retained_payload_bytes=self.resident_bytes(),reserved_bytes=self.reserved_bytes(),
                    resume_reservation_bytes=self.resume_reservation_bytes(),
                    plan=getattr(self,"dense_plan",None),pages=self.dense_pages.receipt() if self.dense_pages is not None else None,
                    qualification="unqualified",physical_release="requires allocator measurement after dropping model anchors")

    def park(self):
        import mlx.core as mx
        if self._parked:return self.dense_receipt()
        if self.dense_pages is not None:self.dense_pages.park()
        mx.synchronize()
        self._resident_resume_bytes=max(self._resident_resume_bytes,self.resident_bytes())
        self._packed.clear();self._dense.clear();self._grouped.clear();self._native_linear_fp32.clear()
        for reader in self._embedding_rows.values():reader.close()
        self._embedding_rows.clear()
        self.file.release_mapping()
        self._parked=True
        return self.dense_receipt()

    def activate(self):
        self.verify_source()
        if self.dense_pages is not None:self.dense_pages.activate()
        self._parked=False
        return self.dense_receipt()

    def array(self, name):
        if self._parked:raise GGUFError("V4.1 weight owner is parked")
        if ".engram_embd." in name:
            raise GGUFError("Engram table may never be materialized")
        if self.dense_pages is not None and name in self.dense_pages.pages:
            if self.tensors[name].ggml_type not in (0,1,24,25,26,27,30):
                raise GGUFError("paged quantized matrices may not be expanded")
            return self.dense_pages.acquire(name)
        return super().array(name)

    def packed(self, name):
        if self._parked:raise GGUFError("V4.1 weight owner is parked")
        if ".engram_embd." in name:
            raise GGUFError("Engram table may never be uploaded")
        if self.expert_stream is not None and self.tensors[name].ggml_type == 19:
            raise GGUFError("streamed expert planes require bounded selected-row gathers")
        if self.dense_pages is not None and name in self.dense_pages.pages:
            if self.tensors[name].ggml_type in (0,1,24,25,26,27,30):
                raise GGUFError("native dense pages require native array access")
            return self.dense_pages.acquire(name)
        return super().packed(name)

    def grouped(self, x, name, *, groups=8):
        if self.dense_pages is not None and name in self.dense_pages.pages:
            self.dense_pages.acquire(name)
        return super().grouped(x,name,groups=groups)

    def close(self):
        if self.dense_pages is not None:self.dense_pages.close()
        super().close()

    def moe(self, x, ids, scores, prefix, *, ticket=None):
        from ..runtime.gguf.segmented_prefill import segmented_prefill_active
        from ..runtime.gguf.v41_metal import moe

        def gather(**kwargs):
            return (self.expert_stream.resolve(ticket, **kwargs) if ticket is not None
                    else self.expert_stream.gather(prefix, ids, **kwargs))

        if (self.expert_stream is not None
                and (x.shape[0] > 1 and not segmented_prefill_active(x.shape[0])
                     and getattr(self, "expert_small_policy", "reference") == "row_exact"
                     or x.shape[0] == 1 and getattr(self, "expert_down_policy", "serial") == "parallel")):
            from ..runtime.gguf.v41_row_exact import moe as row_exact

            ids, planes = gather(row_buffers=True, host_ids=True)
            route = "v41_iq1_row_exact"
            self.scheduled_expert_calls[route] = self.scheduled_expert_calls.get(route, 0)+1
            return row_exact(x, ids, scores, *planes,
                             reuse={"one": 1, "two": 2, "four": 4}[getattr(self, "expert_reuse_policy", "one")],
                             broadcast=getattr(self, "expert_iq1_policy", "scalar") == "broadcast",
                             group_experts=10 if getattr(self,"expert_groups_policy","six") == "ten" else 6,
                             simd_groups=4 if getattr(self,"expert_simd_policy","one") == "four" else 1,
                             counters=self.expert_buffer_counters)

        if (self.expert_stream is not None and self.expert_prefill_policy == "segmented_nax_fp16"
                and getattr(self, "expert_buffer_policy", "stack") == "row_groups"
                and segmented_prefill_active(x.shape[0])):
            from ..runtime.gguf.v41_row_prefill import moe as row_prefill

            ids, planes = gather(row_buffers=True, host_ids=True)
            route = "v41_iq1_segmented_row_buffers_fp16"
            self.scheduled_expert_calls[route] = self.scheduled_expert_calls.get(route, 0)+1
            return row_prefill(x, ids, scores, *planes, counters=self.expert_buffer_counters,
                               group_experts=6 if getattr(self, "expert_binding_policy", "twelve") == "six" else 12)

        if (self.expert_stream is not None and x.shape[0] == 1
                and getattr(self, "expert_decode_policy", "reference") == "row_buffers"):
            from ..runtime.gguf.v41_metal import moe_rows

            ids, planes = gather(row_buffers=True)
            route = "v41_iq1_bf16_row_buffers"
            self.scheduled_expert_calls[route] = self.scheduled_expert_calls.get(route, 0)+1
            return moe_rows(x, ids, scores, *planes)

        if (
            self.expert_prefill_policy == "segmented_nax_fp16"
            and segmented_prefill_active(x.shape[0])
        ):
            from ..runtime.gguf.v41_prefill import moe

        route = (
            "v41_iq1_segmented_nax_fp16"
            if moe.__module__.endswith("v41_prefill")
            else "v41_iq1_bf16_fused"
        )
        self.scheduled_expert_calls[route] = (
            self.scheduled_expert_calls.get(route, 0) + 1
        )
        if self.expert_stream is not None:
            ids, matrices = gather()
        else:
            matrices = tuple(self.packed(prefix + ".ffn_" + kind + "_exps.weight") for kind in ("gate", "up", "down"))
        return moe(
            x,
            ids,
            scores,
            *matrices,
        )

    def can_fuse_dense(self, x, *names):
        from ..runtime.gguf.prefill_scope import dense_prefill_active

        return (self.compute_policy == "cooperative" and not dense_prefill_active(x.size//x.shape[-1])
                and all(self.tensors[name].ggml_type == 8 and len(self.tensors[name].shape) == 2 for name in names))

    def fused_project(self, x, name, *, rounded=True):
        from ..runtime.gguf.v41_dense import project, self_check

        self_check()
        self.scheduled_dense_calls["v41_q8_epilogue"] = self.scheduled_dense_calls.get("v41_q8_epilogue",0)+1
        return project(x,self.packed(name),rounded=rounded)

    def paired_project(self, x, a, b, *, rounded=True):
        from ..runtime.gguf.v41_dense import project, self_check

        self_check()
        self.scheduled_dense_calls["v41_q8_pair"] = self.scheduled_dense_calls.get("v41_q8_pair",0)+1
        return project(x,self.packed(a),self.packed(b),rounded=rounded,rounded_b=rounded)
