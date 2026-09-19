    // Match MLX sdpa_vector_2pass_1 on the compact token order. Splits only
    // distribute the fixed 128 blocks; every block keeps its global index.
    const uint lane = thread_index_in_simdgroup;
    const uint head = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint unit = threadgroup_position_in_grid.z;
    const uint split = unit % S;
    const uint bkv = unit / S;
    const uint b = bkv / NKVH;
    const uint hkv = bkv % NKVH;

    const int L = dims[0];
    const int TOT = dims[1];
    const int U = dims[2];
    const uint count = counts[b * L + row];
    const uint selected = n_sel[b * L + row];
    const int qp = qpos[b * L + row];
    const int complete = ((qp + 1) / BS) * BS;
    const int lpad = left_pad[b];
    const uint base = BLOCKS / S;
    const uint remainder = BLOCKS % S;
    const uint block_begin = split * base + metal::min(split, remainder);
    const uint block_count = base + (split < remainder ? 1u : 0u);
    const uint token_width = uint(U) * BS;
    const uint qh = hkv * GQA + head;
    const uint elements = D / 32;

    float q_values[D / 32];
    for (uint part = 0; part < elements; ++part) {
        const uint d = lane * elements + part;
        const size_t q_index =
            (size_t)b * q_strides[0] +
            (size_t)qh * q_strides[1] +
            (size_t)row * q_strides[2] +
            (size_t)d * q_strides[3];
        q_values[part] = float(scale[0]) * float(q[q_index]);
    }

    const uint slot_base = (b * L + row) * uint(U);
    const size_t k_head =
        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];
    const size_t v_head =
        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];
    for (uint local_block = 0; local_block < block_count; ++local_block) {
        const uint block_idx = block_begin + local_block;
        float out_values[D / 32] = {0};
        float maximum = -3.402823466e+38F;
        float sum = 0.0f;

        for (uint token = block_idx; token < token_width; token += BLOCKS) {
            const uint slot = token / BS;
            const uint tail = token % BS;
            int logical = 0;
            int physical = 0;
            bool live = slot < count;
            if (live) {
                const int block = int(ids[slot_base + slot]);
                logical = block * BS + int(tail);
                physical = lpad + logical;
                live = physical >= 0 && physical < TOT && logical <= qp;
                live = live && (slot < selected || logical >= complete);
                if (HAS_MASK && live) {
                    const uint mask_batch = mask_shape[0] == 1 ? 0 : b;
                    const size_t mask_index =
                        (size_t)mask_batch * mask_strides[0] +
                        (size_t)row * mask_strides[2] +
                        (size_t)physical * mask_strides[3];
                    live = mask[mask_index];
                }
            }
            if (!live) continue;

            float score = 0.0f;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                const size_t k_index =
                    k_head + (size_t)physical * k_strides[2] +
                    (size_t)d * k_strides[3];
                score += q_values[part] * float(k[k_index]);
            }
            score = simd_sum(score);
            const float new_max = metal::max(maximum, score);
            const float factor = fast::exp(maximum - new_max);
            const float probability = fast::exp(score - new_max);
            maximum = new_max;
            sum = sum * factor + probability;
            for (uint part = 0; part < elements; ++part) {
                const uint d = lane * elements + part;
                out_values[part] = out_values[part] * factor
                    + probability * float(
                        v[v_head + (size_t)physical * v_strides[2] +
                          (size_t)d * v_strides[3]]
                    );
            }
        }

        const size_t state = (
            ((size_t)(b * NQH + qh) * L + row) * BLOCKS + block_idx
        );
        if (lane == 0) {
            part_m[state] = maximum;
            part_l[state] = sum;
        }
        for (uint part = 0; part < elements; ++part) {
            const uint d = lane * elements + part;
            part_o[state * D + d] = T(out_values[part]);
        }
    }
    if (unit == 0 && row == 0 && head == 0 && lane == 0)
        engaged[0] = 1;
