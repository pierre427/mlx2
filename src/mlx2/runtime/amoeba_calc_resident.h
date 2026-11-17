// SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
// Copyright (c) 2026 Pierre Lamy
// Bounded, nonrecursive parser shared verbatim by Metal and the CPU test oracle.
// program[0] is instruction count, or negative refusal. Four words per opcode.
#ifndef CALC_NOUNROLL
#define CALC_NOUNROLL _Pragma("clang loop unroll(disable)")
#endif
#define RC_FN __attribute__((noinline))
struct RCJob {
    int begin, end, kind, value, var, bound;
};
RC_FN bool rc_space(int c) { return c == 32 || c == 9 || c == 10 || c == 13; }
RC_FN bool rc_digit(int c) { return c >= '0' && c <= '9'; }
RC_FN int rc_lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
RC_FN bool rc_word(device const uchar *s, int p, int n, constant const char *w) {
    int i = 0;
    CALC_NOUNROLL while (w[i]) {
        if (p + i >= n || rc_lower(s[p + i]) != w[i])
            return false;
        i++;
    }
    return true;
}
RC_FN int rc_skip(device const uchar *s, int p, int n) {
    CALC_NOUNROLL while (p < n && rc_space(s[p])) p++;
    return p;
}

// Low bits of each token class retain the existing number/space boundary law.
// The higher bit admits ordinary, bounded numeric pieces during a generated answer.
// For generated answers, state[1] is the confirmed byte cursor and state[11]
// records whether the single optional leading whitespace piece is still allowed.
enum {
    RC_BOUNDARY_MASK = 3,
    RC_NUMBER_BOUNDARY = 1,
    RC_SPACE_BOUNDARY = 2,
    RC_ANSWER_PIECE = 4,
    RC_FORCE_TOKENS = 0,
    RC_ALLOW_LEADING_SPACE = 1,
    RC_MATCH_ANSWER_BYTES = 2,
};

inline int rc_match_answer_piece(device const int *state, int token,
                                 device const uchar *pool, device const int *offsets,
                                 device const int *lengths, device const int *boundary,
                                 int vocabulary) {
    if (token < 0 || token >= vocabulary || !(boundary[token] & RC_ANSWER_PIECE))
        return -1;
    int start = offsets[token], size = lengths[token], begin = 0, cursor = state[1];
    if (size <= 0 || cursor < 0 || cursor >= state[0])
        return -1;
    if (cursor == 0 && state[11] == RC_ALLOW_LEADING_SPACE) {
        while (begin < size && (pool[start + begin] == ' ' || pool[start + begin] == '\t'))
            begin++;
        if (begin == size)
            return 0;
    }
    if (size - begin > state[0] - cursor)
        return -1;
    for (int i = begin; i < size; i++) {
        int expected = state[16 + cursor + i - begin];
        if (pool[start + i] != pool[offsets[expected]])
            return -1;
    }
    return size - begin;
}
RC_FN int rc_intro(device const uchar *s, int begin, int end) {
    int a = begin;
    if (rc_word(s, a, end, "please "))
        a = rc_skip(s, a + 7, end);
    if (rc_word(s, a, end, "what is "))
        return a + 8;
    if (rc_word(s, a, end, "calculate"))
        a += 9;
    else if (rc_word(s, a, end, "compute"))
        a += 7;
    else if (rc_word(s, a, end, "evaluate"))
        a += 8;
    else if (rc_word(s, a, end, "simplify"))
        a += 8;
    else if (rc_word(s, a, end, "work out"))
        a += 8;
    else if (rc_word(s, a, end, "find the "))
        a += 9;
    else if (rc_word(s, a, end, "give the "))
        a += 9;
    else if (rc_word(s, a, end, "determine the "))
        a += 14;
    else
        return begin;
    if (a >= end || (!rc_space(s[a]) && s[a] != ':' && s[a - 1] != ' '))
        return begin;
    a = rc_skip(s, a, end);
    if (rc_word(s, a, end, "the following "))
        a = rc_skip(s, a + 14, end);
    if (rc_word(s, a, end, "without rounding"))
        a = rc_skip(s, a + 16, end);
    if (rc_word(s, a, end, "exactly"))
        a = rc_skip(s, a + 7, end);
    if (rc_word(s, a, end, "exact "))
        a = rc_skip(s, a + 6, end);
    else if (rc_word(s, a, end, "numerical "))
        a = rc_skip(s, a + 10, end);
    if (rc_word(s, a, end, "value of "))
        a = rc_skip(s, a + 9, end);
    else if (rc_word(s, a, end, "result of "))
        a = rc_skip(s, a + 10, end);
    if (a < end && s[a] == ':')
        a++;
    return a;
}
RC_FN bool rc_ambiguous_date_or_phone(device const uchar *s, int a, int end) {
    int widths[3] = {0, 0, 0};
    int separators[2] = {0, 0};
    int n = 0;
    while (a < end) {
        while (a < end && rc_digit(s[a])) {
            widths[n]++;
            a++;
        }
        if (!widths[n])
            return false;
        if (a == end)
            break;
        if (n == 2 || (s[a] != '-' && s[a] != '/'))
            return false;
        separators[n++] = s[a++];
    }
    if (n == 1)
        return separators[0] == '-' && widths[0] == 3 && widths[1] == 4;
    return n == 2 && separators[0] == separators[1] && widths[0] <= 4 && widths[1] <= 2 &&
           widths[2] <= 4 && widths[2] > 0;
}
RC_FN int rc_column(device const uchar *s, int begin, int end, device uchar *out) {
    const constant char *prefix = "\\begin{array}{r}";
    if (!rc_word(s, begin, end, prefix))
        return 0;
    for (int i = begin; i < end; i++)
        if (s[i] >= 'A' && s[i] <= 'Z')
            return -1;
    int pos = rc_skip(s, begin + 16, end), used = 0;
    for (int row = 0; row < 2; row++) {
        int start = pos;
        if (pos < end && (s[pos] == '+' || s[pos] == '-'))
            out[used++] = s[pos++];
        int digits = 0;
        while (pos < end && (rc_digit(s[pos]) || s[pos] == '.' || s[pos] == ',')) {
            if (used >= 128)
                return -1;
            digits += rc_digit(s[pos]);
            out[used++] = s[pos++];
        }
        if (pos == start || !digits)
            return -1;
        pos = rc_skip(s, pos, end);
        if (pos + 2 > end || s[pos] != '\\' || s[pos + 1] != '\\')
            return -1;
        pos = rc_skip(s, pos + 2, end);
        if (!row) {
            if (!rc_word(s, pos, end, "\\times"))
                return -1;
            pos = rc_skip(s, pos + 6, end);
            out[used++] = '*';
        }
    }
    if (!rc_word(s, pos, end, "\\hline") || rc_skip(s, pos + 6, end) != end)
        return -1;
    return used;
}
RC_FN int rc_group(device const uchar *s, int p, int end) {
    if (p >= end || s[p] != '{')
        return -1;
    int depth = 1;
    CALC_NOUNROLL for (int i = p + 1; i < end; i++) {
        if (s[i] == '{')
            depth++;
        if (s[i] == '}' && !--depth)
            return i;
        if (depth > 48)
            return -1;
    }
    return -1;
}
RC_FN bool rc_bound(device const uchar *s, int a, int end, thread int &value) {
    a = rc_skip(s, a, end);
    int sign = 1;
    if (a < end && (s[a] == '-' || s[a] == '+'))
        sign = s[a++] == '-' ? -1 : 1;
    int n = 0, count = 0;
    CALC_NOUNROLL while (a < end && rc_digit(s[a])) {
        n = n * 10 + s[a++] - '0';
        if (++count > 5)
            return false;
    }
    if (!count || n > 10000 || rc_skip(s, a, end) != end)
        return false;
    value = n * sign;
    return true;
}
RC_FN int rc_prec(int c) {
    return c == '+' || c == '-'               ? 1
           : c == '*' || c == '/' || c == '@' ? 2
           : c == '~'                         ? 3
           : c == '^'                         ? 4
                                              : 0;
}
RC_FN bool rc_emit(device int *p, thread int &n, int op, int a, int b, int c) {
    if (n >= 256)
        return false;
    int i = 1 + 4 * n++;
    p[i] = op;
    p[i + 1] = a;
    p[i + 2] = b;
    p[i + 3] = c;
    return true;
}
RC_FN int rc_op(int c) {
    return c == '+'   ? 1
           : c == '-' ? 2
           : c == '*' ? 3
           : c == '/' ? 4
           : c == '^' ? 5
           : c == '~' ? 6
           : c == '!' ? 7
           : c == '@' ? 8
                      : -1;
}
RC_FN int rc_decimal(int n, device uchar *out) {
    uchar digits[16];
    int k = 0;
    do {
        digits[k++] = uchar('0' + n % 10);
        n /= 10;
    } while (n);
    for (int i = 0; i < k; i++)
        out[i] = digits[k - 1 - i];
    return k;
}

// Deliberately small literal counting grammar. ASCII only, no escapes, no
// inferred lists. List elements must all be signed decimal literals.
RC_FN int rc_count(device const uchar *s, int a, int end, device uchar *out) {
    bool items = false, sum = false;
    if (rc_word(s, a, end, "how many items are in ")) {
        items = true;
        a += 22;
    } else if (rc_word(s, a, end, "how many items in ")) {
        items = true;
        a += 18;
    } else if (rc_word(s, a, end, "count items in ")) {
        items = true;
        a += 15;
    } else if (rc_word(s, a, end, "the sum of ")) {
        sum = true;
        a += 11;
    } else if (rc_word(s, a, end, "sum ")) {
        sum = true;
        a += 4;
    }
    if (items || sum) {
        a = rc_skip(s, a, end);
        if (a >= end || s[a] != '[' || s[end - 1] != ']')
            return -1;
        a++;
        end--;
        int count = 0, used = 0;
        a = rc_skip(s, a, end);
        CALC_NOUNROLL while (a < end) {
            if (++count > 128 || used > 4091)
                return -2;
            if (sum && count > 1)
                out[used++] = '+';
            if (sum)
                out[used++] = '(';
            int digits = 0;
            bool dot = false;
            if (s[a] == '+' || s[a] == '-') {
                if (sum)
                    out[used++] = s[a];
                a++;
            }
            CALC_NOUNROLL while (a < end) {
                int c = s[a];
                if (rc_digit(c)) {
                    digits++;
                } else if (c == '.' && !dot) {
                    dot = true;
                } else
                    break;
                if (used >= 4094)
                    return -2;
                if (sum)
                    out[used++] = s[a];
                a++;
            }
            if (!digits || digits > 60)
                return -1;
            if (sum)
                out[used++] = ')';
            a = rc_skip(s, a, end);
            if (a == end)
                break;
            if (s[a++] != ',')
                return -1;
            a = rc_skip(s, a, end);
            if (a == end)
                return -1;
        }
        return items ? rc_decimal(count, out) : count ? used : rc_decimal(0, out);
    }
    bool how = rc_word(s, a, end, "how many "), count = rc_word(s, a, end, "count ");
    if (!how && !count)
        return 0;
    a += how ? 9 : 6;
    a = rc_skip(s, a, end);
    int target = -1;
    if (a < end && (s[a] == '\'' || s[a] == '"')) {
        int quote = s[a++];
        if (a + 1 >= end || s[a + 1] != quote)
            return -1;
        target = s[a];
        a += 2;
    } else if (a < end)
        target = s[a++];
    else
        return -1;
    if (target < 32 || target >= 127 || target == '\\')
        return -1;
    if (how && a + 1 < end && s[a] == '\'' && s[a + 1] == 's')
        a += 2;
    else if (how && a < end && s[a] == 's')
        a++;
    a = rc_skip(s, a, end);
    if (how && rc_word(s, a, end, "are "))
        a += 4;
    if (!rc_word(s, a, end, "in "))
        return -1;
    a = rc_skip(s, a + 3, end);
    if (rc_word(s, a, end, "the word "))
        a += 9;
    int start = a, stop = a;
    if (a < end && (s[a] == '\'' || s[a] == '"')) {
        int quote = s[a++];
        start = a;
        while (a < end && s[a] != quote) {
            if (s[a] < 32 || s[a] >= 127 || s[a] == '\\')
                return -1;
            a++;
        }
        if (a >= end)
            return -1;
        stop = a++;
    } else {
        start = a;
        while (a < end && ((s[a] >= 'a' && s[a] <= 'z') || (s[a] >= 'A' && s[a] <= 'Z')))
            a++;
        stop = a;
        if (stop == start)
            return -1;
    }
    a = rc_skip(s, a, end);
    bool sensitive = false;
    if (a < end) {
        if (rc_word(s, a, end, "case-sensitive") && a + 14 == end)
            sensitive = true;
        else if ((rc_word(s, a, end, "ignore case") && a + 11 == end) ||
                 (rc_word(s, a, end, "case-insensitive") && a + 16 == end))
            sensitive = false;
        else
            return -1;
    }
    int total = 0;
    for (int i = start; i < stop; i++)
        total += sensitive ? s[i] == target : rc_lower(s[i]) == rc_lower(target);
    return rc_decimal(total, out);
}

// Normalize complete source using an explicit work stack. Fractions become
// parenthesized divisions BEFORE syntax/work bounds are applied. Unknown macros
// cannot be discarded. No suffix salvage, raw text truncation or implicit multiply.
RC_FN int rc_normalize(device const uchar *s, int a, int end, device uchar *out) {
    RCJob jobs[1024];
    int top = 1, used = 0, steps = 0, sizing = 0;
    jobs[0] = {a, end, 0, 0};
    CALC_NOUNROLL while (top && steps++ < 16384) {
        RCJob job = jobs[--top];
        if (job.kind) {
            if (used >= 4096)
                return -2;
            out[used++] = uchar(job.value);
            continue;
        }
        int p = job.begin;
        CALC_NOUNROLL while (p < job.end) {
            int c = s[p++];
            if (c == '@')
                return -1;
            if (rc_space(c)) {
                if (used >= 4096)
                    return -2;
                out[used++] = ' ';
                continue;
            }
            if (job.var && c == job.var) {
                if (used + 10 > 4096)
                    return -2;
                out[used++] = '(';
                int n = job.bound;
                if (n < 0) {
                    out[used++] = '-';
                    n = -n;
                }
                uchar digits[8];
                int count = 0;
                do {
                    digits[count++] = uchar('0' + n % 10);
                    n /= 10;
                } while (n);
                while (count)
                    out[used++] = digits[--count];
                out[used++] = ')';
                continue;
            }
            if (c == '\\') {
                if (p >= job.end)
                    return -1;
                if (s[p] == ',' || s[p] == ';' || s[p] == ':' || s[p] == '!' || s[p] == ' ') {
                    p++;
                    if (used >= 4096)
                        return -2;
                    out[used++] = ' ';
                    continue;
                }
                int start = p;
                while (p < job.end && s[p] >= 'a' && s[p] <= 'z')
                    p++;
                if (p < job.end && s[p] >= 'A' && s[p] <= 'Z')
                    return -1;
                int len = p - start;
                bool frac =
                    (len == 4 && rc_word(s, start, p, "frac")) ||
                    (len == 5 && (rc_word(s, start, p, "dfrac") || rc_word(s, start, p, "tfrac") ||
                                  rc_word(s, start, p, "cfrac")));
                bool binom = (len == 5 && rc_word(s, start, p, "binom")) ||
                             (len == 6 &&
                              (rc_word(s, start, p, "dbinom") || rc_word(s, start, p, "tbinom")));
                bool sum = len == 3 && rc_word(s, start, p, "sum"),
                     prod = len == 4 && rc_word(s, start, p, "prod");
                if (sum || prod) {
                    if (job.var)
                        return -1; // Nested scoped sums are explicitly unsupported here.
                    int q = rc_skip(s, p, job.end);
                    if (q >= job.end || s[q++] != '_')
                        return -1;
                    q = rc_skip(s, q, job.end);
                    int qe = rc_group(s, q, job.end);
                    if (qe < 0)
                        return -1;
                    int v = rc_skip(s, q + 1, qe);
                    if (v >= qe || s[v] < 'a' || s[v] > 'z')
                        return -1;
                    int variable = s[v++];
                    v = rc_skip(s, v, qe);
                    if (v >= qe || s[v++] != '=')
                        return -1;
                    int low = 0, high = 0;
                    if (!rc_bound(s, v, qe, low))
                        return -1;
                    q = rc_skip(s, qe + 1, job.end);
                    if (q >= job.end || s[q++] != '^')
                        return -1;
                    q = rc_skip(s, q, job.end);
                    qe = rc_group(s, q, job.end);
                    if (qe < 0 || !rc_bound(s, q + 1, qe, high))
                        return -1;
                    int body = rc_skip(s, qe + 1, job.end), be = rc_group(s, body, job.end);
                    if (be < 0)
                        return -1;
                    int terms = high >= low ? high - low + 1 : 0;
                    if (terms < 1 || terms > 128 || top + 4 * terms + 3 > 1024)
                        return -2;
                    jobs[top++] = {be + 1, job.end, 0, 0, 0, 0};
                    jobs[top++] = {0, 0, 1, ')', 0, 0};
                    for (int k = high; k >= low; k--) {
                        jobs[top++] = {0, 0, 1, ')'};
                        jobs[top++] = {body + 1, be, 0, 0, variable, k};
                        jobs[top++] = {0, 0, 1, '('};
                        if (k > low)
                            jobs[top++] = {0, 0, 1, sum ? '+' : '*', 0, 0};
                    }
                    if (used >= 4096)
                        return -2;
                    out[used++] = '(';
                    break;
                }
                if (frac || binom) {
                    int x = rc_skip(s, p, job.end), xe = rc_group(s, x, job.end);
                    if (xe < 0)
                        return -1;
                    int y = rc_skip(s, xe + 1, job.end), ye = rc_group(s, y, job.end);
                    if (ye < 0)
                        return -1;
                    if (top + 9 > 1024)
                        return -2;
                    jobs[top++] = {ye + 1, job.end, 0, 0, job.var, job.bound};
                    jobs[top++] = {0, 0, 1, ')'};
                    jobs[top++] = {0, 0, 1, ')'};
                    jobs[top++] = {y + 1, ye, 0, 0, job.var, job.bound};
                    jobs[top++] = {0, 0, 1, '('};
                    jobs[top++] = {0, 0, 1, frac ? '/' : '@'};
                    jobs[top++] = {0, 0, 1, ')'};
                    jobs[top++] = {x + 1, xe, 0, 0, job.var, job.bound};
                    jobs[top++] = {0, 0, 1, '('};
                    if (used >= 4096)
                        return -2;
                    out[used++] = '(';
                    break;
                }
                if (len == 4 && rc_word(s, start, p, "left")) {
                    int q = rc_skip(s, p, job.end);
                    if (q >= job.end || (s[q] != '(' && s[q] != '{'))
                        return -1;
                    sizing++;
                    continue;
                }
                if (len == 5 && rc_word(s, start, p, "right")) {
                    int q = rc_skip(s, p, job.end);
                    if (q >= job.end || (s[q] != ')' && s[q] != '}'))
                        return -1;
                    if (--sizing < 0)
                        return -1;
                    continue;
                }
                if ((len == 3 && rc_word(s, start, p, "big")) ||
                    (len == 4 && (rc_word(s, start, p, "bigl") || rc_word(s, start, p, "bigr"))) ||
                    (len == 4 && rc_word(s, start, p, "quad")) ||
                    (len == 5 && rc_word(s, start, p, "qquad")) ||
                    (len == 12 && rc_word(s, start, p, "displaystyle")) ||
                    (len == 9 && rc_word(s, start, p, "textstyle"))) {
                    if (used >= 4096)
                        return -2;
                    out[used++] = ' ';
                    continue;
                }
                if ((len == 5 && rc_word(s, start, p, "times")) ||
                    (len == 4 && rc_word(s, start, p, "cdot")))
                    c = '*';
                else if (len == 3 && rc_word(s, start, p, "div"))
                    c = '/';
                else
                    return -1;
            } else if (p > job.begin + 1 && rc_space(s[p - 2])) {
                int start = p - 1, word = 0, op = 0;
                if (rc_word(s, start, job.end, "times")) {
                    word = 5;
                    op = '*';
                } else if (rc_word(s, start, job.end, "multiplied by")) {
                    word = 13;
                    op = '*';
                } else if (rc_word(s, start, job.end, "divided by")) {
                    word = 10;
                    op = '/';
                } else if (rc_word(s, start, job.end, "plus")) {
                    word = 4;
                    op = '+';
                } else if (rc_word(s, start, job.end, "minus")) {
                    word = 5;
                    op = '-';
                }
                if (word && start + word < job.end && rc_space(s[start + word])) {
                    c = op;
                    p = start + word;
                } else if (c == 0xc3 && p < job.end && (s[p] == 0x97 || s[p] == 0xb7))
                    c = s[p++] == 0x97 ? '*' : '/';
                else if (c == 0xe2 && p + 1 < job.end && s[p] == 0x88 && s[p + 1] == 0x92) {
                    p += 2;
                    c = '-';
                }
            } else if (c == 0xc3 && p < job.end && (s[p] == 0x97 || s[p] == 0xb7)) {
                c = s[p++] == 0x97 ? '*' : '/';
            } else if (c == 0xe2 && p + 1 < job.end && s[p] == 0x88 && s[p + 1] == 0x92) {
                p += 2;
                c = '-';
            }
            if (used >= 4096)
                return -2;
            out[used++] = uchar(c);
        }
    }
    return top ? -2 : sizing ? -1 : used;
}

RC_FN void resident_parse(device const uchar *source, int length, int mode, device int *program,
                          device uchar *normalized) {
    program[0] = -1;
    if (length < 1 || length > 4096)
        return;
    int a = rc_skip(source, 0, length), end = length;
    CALC_NOUNROLL while (end > a && rc_space(source[end - 1])) end--;
    // Only fixed instruction wrappers, never arbitrary prose before a formula.
    bool explicit_math = false;
    if (mode == 0) {
        int expression = rc_intro(source, a, end);
        explicit_math = expression > a;
        a = expression;
        if (end > a && (source[end - 1] == '?' || source[end - 1] == '.'))
            end--;
    }
    a = rc_skip(source, a, end);
    CALC_NOUNROLL while (end > a && rc_space(source[end - 1])) end--;
    // A leading dash followed by horizontal space is a Markdown bullet on a
    // generated line. Direct requests and multi-line column math keep their signs.
    if (mode == 1 && end - a >= 2 && source[a] == '-' &&
        (source[a + 1] == ' ' || source[a + 1] == '\t')) {
        int line_end = a + 2;
        CALC_NOUNROLL while (line_end < end && source[line_end] != '\n' && source[line_end] != '\r')
            line_end++;
        if (line_end == end)
            a = rc_skip(source, a + 2, end);
    }
    // Balanced outer math delimiters. Trailing '=\\)' is deliberately supported.
    if (end - a >= 4 && source[a] == '\\' && (source[a + 1] == '(' || source[a + 1] == '[')) {
        int close = source[a + 1] == '(' ? ')' : ']';
        if (source[end - 2] != '\\' || source[end - 1] != close)
            return;
        a += 2;
        end -= 2;
        explicit_math = true;
    } else if (end - a >= 2 && source[a] == '$') {
        if (source[end - 1] != '$')
            return;
        a++;
        end--;
        explicit_math = true;
        if (a < end && source[a] == '$') {
            if (source[end - 1] != '$')
                return;
            a++;
            end--;
        }
    }
    CALC_NOUNROLL while (end > a && rc_space(source[end - 1])) end--;
    int len = rc_column(source, a, end, normalized);
    bool operation = false;
    if (!len) {
        if (end > a && source[end - 1] == '=')
            end--;
        else if (mode == 1) {
            if (end - a > 7 && rc_word(source, end - 6, end, "equals") && rc_space(source[end - 7]))
                end -= 6;
            else
                return;
        }
        while (end > a && rc_space(source[end - 1]))
            end--;
        if (!explicit_math && rc_ambiguous_date_or_phone(source, a, end))
            return;
        len = rc_count(source, a, end, normalized);
        operation = len > 0;
        if (!len)
            len = rc_normalize(source, a, end, normalized);
    }
    if (len <= 0) {
        program[0] = len ? len : -1;
        return;
    }
    int ops[96], top = 0, n = 0, depth = 0;
    bool want = true;
    CALC_NOUNROLL for (int i = 0; i < len;) {
        int c = normalized[i];
        if (rc_space(c)) {
            i++;
            continue;
        }
        if (rc_digit(c) || c == '.') {
            if (!want)
                return;
            int start = i, digits = 0, scale = 0, group = 0, groups = 0;
            bool decimal = false;
            CALC_NOUNROLL while (i < len) {
                c = normalized[i];
                if (rc_digit(c)) {
                    digits++;
                    if (decimal)
                        scale++;
                    else
                        group++;
                    i++;
                    continue;
                }
                if (c == '.' && !decimal) {
                    if (groups && group != 3)
                        return;
                    decimal = true;
                    i++;
                    continue;
                }
                if (c == ',' && !decimal) {
                    if (!group || (groups ? group != 3 : group > 3))
                        return;
                    groups++;
                    group = 0;
                    i++;
                    continue;
                }
                break;
            }
            if (!digits || digits > 60 || scale > 60 || (groups && !decimal && group != 3))
                return;
            if (!rc_emit(program, n, 0, start, i, scale)) {
                program[0] = -2;
                return;
            }
            want = false;
            continue;
        }
        i++;
        if (c == '(' || c == '{') {
            if (!want || top >= 96 || ++depth > 48)
                return;
            ops[top++] = c;
            continue;
        }
        if (c == ')' || c == '}') {
            if (want)
                return;
            int match = c == ')' ? '(' : '{';
            CALC_NOUNROLL while (
                top && ops[top - 1] != '(' &&
                ops[top - 1] != '{') if (!rc_emit(program, n, rc_op(ops[--top]), 0, 0, 0)) return;
            if (!top || ops[--top] != match)
                return;
            depth--;
            want = false;
            continue;
        }
        if (c == '!') {
            int next = rc_skip(normalized, i, len);
            if (next < len && normalized[next] == '!')
                return;
            if (want || !rc_emit(program, n, 7, 0, 0, 0))
                return;
            operation = true;
            continue;
        }
        if (want) {
            if (c == '+')
                continue;
            if (c == '-') {
                if (top >= 96)
                    return;
                ops[top++] = '~';
                continue;
            }
            return;
        }
        if (!rc_prec(c) || c == '~')
            return;
        operation = true;
        CALC_NOUNROLL while (top && rc_prec(ops[top - 1]) &&
                             (rc_prec(ops[top - 1]) > rc_prec(c) ||
                              (c != '^' && rc_prec(ops[top - 1]) == rc_prec(c)))) {
            if (!rc_emit(program, n, rc_op(ops[--top]), 0, 0, 0))
                return;
        }
        if (top >= 96)
            return;
        ops[top++] = c;
        want = true;
    }
    if (want || depth || !operation)
        return;
    CALC_NOUNROLL while (top) if (!rc_emit(program, n, rc_op(ops[--top]), 0, 0, 0)) return;
    program[0] = n;
}

#ifdef RESIDENT_ARITHMETIC
RC_FN Rat rc_integer(int n) {
    Rat r = one_d();
    r.n = zero_b();
    r.sign = n ? 1 : 0;
    int i = 0;
    CALC_NOUNROLL while (n) {
        r.n.d[i++] = uchar(n % 10);
        n /= 10;
    }
    return r;
}
RC_FN int rc_small(Rat a, int maximum, thread int &err) {
    if (a.sign < 0 || cmp(a.den, one_b()) != 0 || width(a.n) > 3) {
        err = 4;
        return 0;
    }
    int n = a.n.d[0] + 10 * a.n.d[1] + 100 * a.n.d[2];
    if (n > maximum)
        err = 4;
    return n;
}
RC_FN Rat rc_factorial(Rat a, thread int &err) {
    int n = rc_small(a, 100, err);
    Rat r = one_d();
    CALC_NOUNROLL for (int i = 2; i <= n && !err; i++) r = norm(mul_d(r, rc_integer(i), err), err);
    return r;
}
RC_FN Rat rc_choose(Rat a, Rat b, thread int &err) {
    int n = rc_small(a, 256, err), k = rc_small(b, 256, err);
    if (k > n)
        err = 4;
    k = k < n - k ? k : n - k;
    Rat r = one_d();
    CALC_NOUNROLL for (int i = 1; i <= k && !err; i++) {
        Rat v = rc_integer(n - k + i);
        v.den = rc_integer(i).n;
        r = norm(mul_d(r, v, err), err);
    }
    return r;
}
RC_FN void resident_evaluate(device const int *program, device const uchar *text,
                             device int *result) {
    int n = program[0], err = 0, top = 0;
    result[0] = 7;
    CALC_NOUNROLL for (int j = 1; j < 268; j++) result[j] = 0;
    if (n < 1 || n > 256)
        return;
    Rat stack[64];
    CALC_NOUNROLL for (int pc = 0; pc < n && !err; pc++) {
        int p = 1 + 4 * pc, op = program[p];
        Rat a, b, r;
        if (op == 0) {
            if (top >= 64) {
                err = 7;
                break;
            }
            r = one_d();
            r.n = zero_b();
            int start = program[p + 1], end = program[p + 2], scale = program[p + 3], digit = 0;
            if (start < 0 || end > 4096 || start >= end || scale < 0 || scale > 60) {
                err = 7;
                break;
            }
            CALC_NOUNROLL for (int i = end - 1; i >= start; i--) if (rc_digit(text[i])) {
                if (digit >= 60) {
                    err = 1;
                    break;
                }
                r.n.d[digit++] = text[i] - '0';
            }
            r.den = shift_b(one_b(), scale, err);
        } else if (op == 6 || op == 7) {
            if (top < 1) {
                err = 7;
                break;
            }
            a = stack[--top];
            if (op == 6) {
                r = a;
                r.sign = -r.sign;
            } else
                r = rc_factorial(a, err);
        } else {
            if (top < 2) {
                err = 7;
                break;
            }
            b = stack[--top];
            a = stack[--top];
            if (op == 1 || op == 2)
                r = add_d(a, b, op == 2, err);
            else if (op == 3)
                r = mul_d(a, b, err);
            else if (op == 4) {
                Big temp = b.n;
                b.n = b.den;
                b.den = temp;
                r = mul_d(a, b, err);
            } else if (op == 5)
                r = pow_d(a, b, err);
            else if (op == 8)
                r = rc_choose(a, b, err);
            else {
                err = 7;
                break;
            }
        }
        if (!err)
            r = norm(r, err);
        if (!err)
            stack[top++] = r;
    }
    if (!err && top != 1)
        err = 7;
    if (!err)
        emit(stack[0], result, err);
    else
        result[0] = err;
}
#endif
