# Q5_1 CPU reference check

Checked against upstream main `c6552692def54da2cd854c6b4a6f5fb4ebe6ab3b`.
`dequant_row` has no Q5_1 branch; its generic loop advances by 256 values.
Short rows can be left unwritten, and complete superblocks do not have a valid
Q5_1 decoder either. The independent fixture fails immediately at K=32:
`got=-123456 expected=-0.5`.

Decode each 32-value block using its F16 scale, F16 minimum, 32 high bits and
16 packed nibble bytes. This also handles all 640 values of Flash Next's expert
down rows, including their final 128 values.

After the fix, 3,424 values across K=32/64/128/256/512/640/768/1024 match the
independent encoder's expected values exactly. Leading/trailing output guards
remain unchanged. AddressSanitizer/UndefinedBehaviorSanitizer also pass, with
leak detection disabled because the diagnostic environment uses ptrace.

Native Linux `runz` and `serverz` build. `ref_forward` also builds with the
temporary compiler alias `-D_fseeki64=fseeko`; its existing Windows seek spelling
is separate from this decoder fix. Windows tests and current-main model KL
measurements were not run. The production B70 server was left running.

```sh
g++ -std=c++20 -O2 -mavx2 -mfma -mf16c tests/test_q51_reference.cpp -o /tmp/test_q51_reference
/tmp/test_q51_reference
```

This repairs reference values, with no GPU kernel/layout or speed claim. A
corrected reference can change reported KL; it does not itself validate GPU
arithmetic. Downstream AutoRound benchmarks use a different expert quant and
are a separate submission.
