// Tail-biting trellis Viterbi for EXL3-format 16x16 tiles (original mlx2 code).
//
// A tile is 256 slots in decode order. Slot s reads the codeword made of the
// last `w` stream bits ending at bit E(s); between slots the stream advances
// by k_s = E(s) - E(s-1) bits (k_0 = E(0)), so the state is the w-bit window
// and a step shifts k_s new bits in: S' = ((S << k) | b) & (2^w - 1). The
// stream is circular (slot 0's window wraps into the end of the stream).
//
// Pass 1 runs a free-start/free-end Viterbi over a window of `probe` slots
// either side of the wrap point (slots 256-probe .. 255, 0 .. probe-1) and
// reads the overlap bits O (low w-k_0 bits of slot 255's state). Pass 2 runs
// the whole tile in order with slot 0 restricted to states whose top w-k_0
// bits are O and slot 255 restricted to states whose low w-k_0 bits are O, so
// the result is always a valid circular stream (the QTIP tail-biting
// approximation, shortened to a window around the wrap).
//
// Cost: plain squared error to the slot targets. Memory per call: two cost
// arrays of 2^w floats and 256 * 2^(w-kmin) backpointer bytes.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BIG 3.0e38f

static int run_pass(const float *cb, int w, const uint8_t *shifts, const int *order, int len,
                    const float *target, int start_o, int end_o, float *costa, float *costb,
                    uint8_t *bp, float *mbuf, uint16_t *states_out) {
  const int nstate = 1 << w;
  const uint32_t mask = (uint32_t)nstate - 1;
  // Step 0.
  int s0 = order[0];
  float t = target[s0];
  int k0 = shifts[s0];
  for (int S = 0; S < nstate; S++) {
    float d = cb[S] - t;
    int ok = start_o < 0 || (S >> k0) == start_o;
    costa[S] = ok ? d * d : BIG;
  }
  size_t bpoff = 0;
  size_t offs[256];
  for (int i = 1; i < len; i++) {
    int s = order[i];
    int k = shifts[s];
    int ng = nstate >> k;
    int nj = 1 << k;
    t = target[s];
    uint8_t *b = bp + bpoff;
    offs[i] = bpoff;
    bpoff += (size_t)ng;
    // Min over the 2^k predecessors sharing g = S >> k; first index wins
    // ties. (A j-outer vectorizable layout measured 3x slower on M5.)
    for (int g = 0; g < ng; g++) {
      float m = costa[g];
      int arg = 0;
      for (int j = 1; j < nj; j++) {
        float v = costa[(j << (w - k)) | g];
        if (v < m) { m = v; arg = j; }
      }
      b[g] = (uint8_t)arg;
      int base = g << k;
      for (int lo = 0; lo < nj; lo++) {
        float d = cb[base | lo] - t;
        costb[base | lo] = m + d * d;
      }
    }
    float *tmp = costa; costa = costb; costb = tmp;
  }
  int last = order[len - 1];
  (void)last;
  float best = BIG;
  int bs = 0;
  uint32_t omask = (1u << (w - shifts[order[0]])) - 1;
  for (int S = 0; S < nstate; S++) {
    if (end_o >= 0 && (int)(S & omask) != end_o) continue;
    if (costa[S] < best) { best = costa[S]; bs = S; }
  }
  uint32_t S = (uint32_t)bs;
  states_out[len - 1] = (uint16_t)S;
  for (int i = len - 1; i >= 1; i--) {
    int k = shifts[order[i]];
    uint32_t g = S >> k;
    uint32_t j = bp[offs[i] + g];
    S = ((j << (w - k)) | g) & mask;
    states_out[i - 1] = (uint16_t)S;
  }
  return 0;
}

// targets [n_tiles, 256] in slot order; states out [n_tiles, 256] (slot order).
// cb: codebook value per w-bit state. Returns 0 on success.
int viterbi_tiles(const float *targets, int n_tiles, const float *cb, int w,
                  const uint8_t *shifts, int probe, uint16_t *states) {
  const int nstate = 1 << w;
  int kmin = 16;
  for (int s = 0; s < 256; s++) if (shifts[s] < kmin) kmin = shifts[s];
  if (kmin < 1 || w > 16 || probe < 1 || probe > 128) return -1;
  float *costa = malloc(sizeof(float) * nstate);
  float *costb = malloc(sizeof(float) * nstate);
  uint8_t *bp = malloc((size_t)256 * (nstate >> kmin));
  float *mbuf = malloc(sizeof(float) * (nstate >> kmin));
  if (!costa || !costb || !bp || !mbuf) { free(costa); free(costb); free(bp); free(mbuf); return -2; }
  int order1[256], order2[256];
  uint16_t st1[256];
  int len1 = 2 * probe;
  for (int i = 0; i < probe; i++) { order1[i] = 256 - probe + i; order1[probe + i] = i; }
  for (int i = 0; i < 256; i++) order2[i] = i;
  int k0 = shifts[0];
  uint32_t omask = (1u << (w - k0)) - 1;
  for (int tile = 0; tile < n_tiles; tile++) {
    const float *t = targets + (size_t)tile * 256;
    run_pass(cb, w, shifts, order1, len1, t, -1, -1, costa, costb, bp, mbuf, st1);
    int o = (int)(st1[probe - 1] & omask);  // slot 255's state
    run_pass(cb, w, shifts, order2, 256, t, o, o, costa, costb, bp, mbuf, states + (size_t)tile * 256);
  }
  free(costa); free(costb); free(bp); free(mbuf);
  return 0;
}
