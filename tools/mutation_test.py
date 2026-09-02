"""Mutation test for the chi = 2 zipper contraction.

Perturbs each of the eight product terms of `TensorNetworkUnit::zipper2_step`
in turn by 5%% and checks that something notices. A term whose mutant survives
everything is a term nothing is actually testing.

Run from the repository root:

    python tools/mutation_test.py

Last run: 8/8 caught by `cargo test`, 6/8 by programs/mps_ghz_overlap.uasm
alone. The two the program misses (stage1 im: e_im*b_re, stage2 re:
a_im*t_im) vanish for that particular pair of states; the random-MPS ablation
in the unit tests covers them.

The file is restored from a backup in a finally block, so an interrupted run
does not leave a mutated source behind.
"""
import io, subprocess, shutil, sys

SRC = 'src/alu.rs'
BAK = SRC + '.mutbak'

# The eight product terms of the two contraction stages.
TERMS = [
    ("stage1 re: e_re*b_re", "re += e_re * b_re - e_im * b_im;",
                             "re += e_re * b_re * 1.05 - e_im * b_im;"),
    ("stage1 re: e_im*b_im", "re += e_re * b_re - e_im * b_im;",
                             "re += e_re * b_re - e_im * b_im * 1.05;"),
    ("stage1 im: e_re*b_im", "im += e_re * b_im + e_im * b_re;",
                             "im += e_re * b_im * 1.05 + e_im * b_re;"),
    ("stage1 im: e_im*b_re", "im += e_re * b_im + e_im * b_re;",
                             "im += e_re * b_im + e_im * b_re * 1.05;"),
    ("stage2 re: a_re*t_re", "re += a_re * t_re + a_im * t_im;",
                             "re += a_re * t_re * 1.05 + a_im * t_im;"),
    ("stage2 re: a_im*t_im", "re += a_re * t_re + a_im * t_im;",
                             "re += a_re * t_re + a_im * t_im * 1.05;"),
    ("stage2 im: a_re*t_im", "im += a_re * t_im - a_im * t_re;",
                             "im += a_re * t_im * 1.05 - a_im * t_re;"),
    ("stage2 im: a_im*t_re", "im += a_re * t_im - a_im * t_re;",
                             "im += a_re * t_im - a_im * t_re * 1.05;"),
]

shutil.copy(SRC, BAK)
original = io.open(BAK, encoding='utf-8').read()
caught_prog, caught_suite, survived = 0, 0, []

try:
    for name, find, repl in TERMS:
        assert find in original, "term not found: " + name
        io.open(SRC, 'w', encoding='utf-8').write(original.replace(find, repl, 1))

        build = subprocess.run(['cargo', 'build', '--release'],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        if build.returncode != 0:
            print("  %-26s BUILD FAILED" % name)
            continue

        run = subprocess.run(['./target/release/unibit.exe', 'run',
                              'programs/mps_ghz_overlap.uasm'],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        prog_caught = 'FAIL' in run.stdout

        suite = subprocess.run(['cargo', 'test', '--quiet'],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        suite_caught = suite.returncode != 0

        if prog_caught:
            caught_prog += 1
        if suite_caught:
            caught_suite += 1
        if not prog_caught and not suite_caught:
            survived.append(name)

        print("  %-26s program:%s  suite:%s" % (
            name,
            "CAUGHT " if prog_caught else "survived",
            "CAUGHT" if suite_caught else "survived"))
finally:
    shutil.copy(BAK, SRC)
    import os
    os.remove(BAK)
    subprocess.run(['cargo', 'build', '--release'], capture_output=True)

print("\n%d/%d mutants caught by the .uasm program" % (caught_prog, len(TERMS)))
print("%d/%d mutants caught by the test suite" % (caught_suite, len(TERMS)))
if survived:
    print("SURVIVED EVERYTHING (untested code): " + ", ".join(survived))
    sys.exit(1)
print("no mutant survives both")
