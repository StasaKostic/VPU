#!/usr/bin/env python3
# Lays out A of a gemm32 application offline, like HEEPtor's generator, so
# that the kernel spends no cycles on it:
#   a_t.h:          A transposed and scaled, a_t[k * GEMM_SIZE + m], for the
#                   tail of the kernel, if the VLEN leaves one
#   gemm_kernel.h:  (--kernel) the head of the kernel with A in its
#                   instructions, as li immediates of the vmulh.vx scalars
# Usage: gen_gemm32_a.py [--kernel] [--vlen N] sw/applications/<app>
import argparse
import re
from pathlib import Path

# Must match BLOCK (ROW_BLOCK) in main.c.
BLOCK = 4


def read_data(path):
    text = path.read_text()
    size = int(re.search(r"#define GEMM_SIZE (\d+)", text).group(1))
    bits = int(re.search(r"#define GEMM_INPUT_BITS (\d+)", text).group(1))
    body = re.search(r"scalar_a\[\d+\][^=]*=\s*\{(.*?)\};", text, re.S).group(1)
    values = [int(v) for v in re.findall(r"-?\d+", body)]
    assert len(values) == size * size, "scalar_a must hold GEMM_SIZE^2 values"
    return size, bits, [values[m * size:(m + 1) * size] for m in range(size)]


def write_table(out, size, width, a_scaled):
    rows = []
    for k in range(size):
        rows.append("    " + ", ".join(str(a_scaled[m][k]) for m in range(size)) + ",")
    out.write_text(
        "// Generated from data.h by scripts/gen_gemm32_a.py -- do not edit.\n"
        "// A transposed and already scaled, a_t[k * GEMM_SIZE + m] =\n"
        "// A[m][k] << A_SCALE_SHIFT, so that column k of A is a contiguous strip.\n"
        "// It is not const: a const array cannot share .xheep_data_interleaved\n"
        "// with the writable arrays of main.c.\n"
        "#pragma once\n\n"
        f"static int{width * 2}_t a_t[GEMM_SIZE * GEMM_SIZE] "
        "__attribute__((section(\".xheep_data_interleaved\"))) = {\n"
        + "\n".join(rows) + "\n};\n")


def head_pairs(size, width, vlen):
    # Column pairs of a row of C in full vectors: HEAD_PAIRS in main.c, all
    # HALF_SIZE pairs when the vector length divides them.
    vec_elems = vlen // (width * 2)
    return size // 2 // vec_elems * vec_elems


def write_kernel(out, size, width, vlen, a_scaled):
    sew = width * 2
    elem_bytes = sew // 8
    half = size // 2
    vec_elems = vlen // sew
    # Row stride of the results of the head: HEAD_PAIRS, or HALF_SIZE without a tail.
    row_pairs = head_pairs(size, width, vlen)
    lines = []
    for m in range(0, size, BLOCK):
        lines.append(f"# rows {m}..{m + BLOCK - 1}")
        lines.append("mv t4, %0")
        for k in range(size):
            if k:
                lines.append(f"addi t4, t4, {half * elem_bytes}")
            lines.append(f"vle{sew}.v v0, (t4)")
            for r in range(BLOCK):
                lines.append(f"li t{r}, {a_scaled[m + r][k]}")
                if k == 0:
                    lines.append(f"vmulh.vx v{2 + r}, v0, t{r}")
                else:
                    lines.append(f"vmulh.vx v1, v0, t{r}")
                    lines.append(f"vadd.vv v{2 + r}, v{2 + r}, v1")
        for r in range(BLOCK):
            lines.append(f"addi t5, %1, {(m + r) * row_pairs * elem_bytes}")
            lines.append(f"vse{sew}.v v{2 + r}, (t5)")
    asm = "\n".join(f'        "{line}\\n\\t"' for line in lines)
    out.write_text(
        "// Generated from data.h by scripts/gen_gemm32_a.py -- do not edit.\n"
        "// Head of gemm_vfu with A in its instructions, like HEEPtor's MUL\n"
        "// immediates: RVV has no multiply by an immediate, so each scaled\n"
        "// A[m][k] is an li into the scalar register of vmulh.vx. One call\n"
        f"// computes one strip of {vec_elems} pairs of one parity for all rows of C:\n"
        "//   b: b_even or b_odd at the first pair of the strip\n"
        f"//   c: c_even or c_odd at the first pair of the strip, row stride {row_pairs} pairs\n"
        "// The caller sets vl to VEC_ELEMS.\n"
        "#pragma once\n\n"
        "static void __attribute__((noipa, optimize(\"no-tree-vectorize\"))) "
        "gemm_head_strip(const acc_t *b, acc_t *c)\n"
        "{\n"
        "    asm volatile(\n"
        + asm + "\n"
        "        :: \"r\"(b), \"r\"(c)\n"
        "        : \"t0\", \"t1\", \"t2\", \"t3\", \"t4\", \"t5\",\n"
        "          \"v0\", \"v1\", \"v2\", \"v3\", \"v4\", \"v5\", \"memory\");\n"
        "}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path, help="application directory with data.h")
    parser.add_argument("--kernel", action="store_true",
                        help="also generate gemm_kernel.h with A in the instructions")
    parser.add_argument("--vlen", type=int, default=192,
                        help="VLEN_BITS of main.c (default: 192)")
    args = parser.parse_args()

    size, bits, a = read_data(args.app / "data.h")
    # Same container and scale as main.c: INPUT_WIDTH and A_SCALE_SHIFT.
    width = 8 if bits <= 8 else 16
    shift = width + 1 - bits
    a_scaled = [[v << shift for v in row] for row in a]
    # With A in the instructions, only a tail loads A from memory.
    if not args.kernel or head_pairs(size, width, args.vlen) < size // 2:
        write_table(args.app / "a_t.h", size, width, a_scaled)
    if args.kernel:
        write_kernel(args.app / "gemm_kernel.h", size, width, args.vlen, a_scaled)


if __name__ == "__main__":
    main()
