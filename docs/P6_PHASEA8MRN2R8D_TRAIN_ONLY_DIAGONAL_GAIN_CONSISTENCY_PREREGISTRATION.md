# P6 Phase-A8MRN2R8D Train-only Diagonal Gain-consistency Preregistration

## 结论

终态分类：`P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTERED`；`R8E_execution_ready=true`。唯一 H4 candidate 为 `H4_H3_TRAIN140_DENSE_UNIT_DIAGONAL_CONSISTENCY_LAMBDA0P1`，parent=H3。本轮只完成预注册与预演，没有训练 H4。

## R8C 硬冻结与 H3 结果

R8C 报告和 17 个 JSON 已与原 20 文件提交包逐一 SHA256 核对，`status=FINALIZED`、20000 updates、CASE B、`CANCEL3=false`、`READY3=false`、field stream match、fresh30=0/0、operator=0、new physics=0 全部通过。冻结后 R8C 保持只读。

H3 best update=18000，VAL pooled/max=0.0532086981787/0.0786813591841；final VAL pooled/max=0.0533076989738/0.0903888077648。INTERNAL full median/max=0.0584776383788/0.0708409221401，energy=0.0175789245803/0.0424591538883，boundary=0.0586053254967/0.0704064661661，gain max=0.0359256027823，peak max=0。rawA/rawB median=0.0615027651417/0.0610168919488，error cosine=0.807380882978，cancellation ratio=0.950650039116，`CANCEL3=false`。

burned24 full median/max=0.0666017957809/0.121394772363，energy=0.0178803822119/0.0549902415093，boundary=0.0663418633078/0.126313528705，gain=0.0705224080657/0.187211550575，peak max=0。H3 相对 H2 的 full median/max 改善=0.015644312955%/-7.57852040335%，boundary median/max=0.926655224665%/-8.95856862373%，interior=-4.24761097642%，corner=-10.2260097289%，corner/interior ratio=-5.7348064828%，energy median/max=46.2476883858%/33.002194978%，gain median/max=-71.2426214865%/-26.0831681277%。

H3 已明显改善 energy，且 boundary/peak gate 通过，但 full median 仅改善 0.0156%，full max、corner 和 gain 反而变差。因此不继续 boundary-feature sweep；下一步只隔离检验 TRAIN140-derived diagonal/gain consistency。

## TRAIN140-only 对角先验

TRAIN140 normalized real diagonal: n=140，min/median/max=0.997901328207/1/1.0039199728，mean/std=1.00030414041/0.0013447644067，span=0.00601864459298，max|deviation from 1|=0.00391997279957，imag max=0。这只支持 `AN_APPROXIMATELY_CONSTANT_NORMALIZED_DIAGONAL_PRIOR`；不声称每个 native source 数学上精确等于 1。target=`1+0j`，唯一来源为 TRAIN140 median=1.0；VAL20、INTERNAL_TEST20、burned24 均未参与 prior 定义。

## H4 唯一变更与 loss

H4 完全等于 H3 architecture，保持 414210 参数、165 维输入、boundary features/source masks/output PE/relative PE、model seed=20260909 与 fresh initial-state SHA `7c4601d5c779655bc2a69482995868aa501310f5de3d1a12470708202424bb28`。不 warm-start H3/H2 checkpoint。

每 update 仍为 4个 TRAIN source x 8192 query=32768 field pairs。对每个 source 块用固定 local indices `0,32,...,8160` 从 post-C4 safe query 中选 256 个，合计 1024 q/update；不使用新 RNG，不根据 target error 选 q。构造 `(q,q,d=0)`：

`L_field = 0.5*MSE_complex(A,y) + 0.5*MSE_complex(B,y)`

`L_diag = 0.5*MSE_complex(A_diag,1+0j) + 0.5*MSE_complex(B_diag,1+0j)`

`L_total = L_field + 0.1*L_diag`

field coefficient 仍为 1.0，`lambda_diag=0.1` 在首次 science optimizer step 前冻结；不声称 0.1 最优，不做 sweep。对 A_diag 和 B_diag 分别监督，不只监督最终平均，以避免 raw-branch underdetermination。

## 坐标流、安全性与等价性

20k 纯坐标复演共审计 20480000 个 diagonal q，unique=2698300，x/y/z levels=150/150/120，boundary/interior occurrences=16952682/3527318，前后 C4 field/diagonal blacklist hits 均为 0。未打开 PSF 数组，未保存巨大 stream。field stream SHA 仍为 `6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2`；diagonal stream SHA 为 `395bd3a0e85405db340aab23a0fce4f334b95bf6750b029bbb307b399db526ce`。

H4 相对 H3 在 4096 对坐标的最大归一化 forward discrepancy=0（门限 1e-7），initial SHA identity PASS，Hermitian discrepancy=0（门限 1e-6），field/diag gradient 非零，no detach。inference 仍为 `0.5*(A+B)`，禁止 hard-code center、column rescaling、post-hoc normalization、gain clipping 或手工 amplitude correction。

## 未来 R8E 训练、selection 与 outcome

split=140/20/20，split SHA=`29900f4738d26f36893d3ebb5aae795b32d5836888a267c4bee98d440372a9ac`，blacklist unique=136。AdamW lr=2e-4、betas=(0.9,0.99)、weight decay=1e-6、scheduler=null、20000 updates、VAL/checkpoint every500、no early stopping 全部不变。VAL 仅用 20 sources x 65536 probes，coordinate SHA=`718c4152461367368be80e02408c18667da09b18dc5550327337ccc06c8fb24b`；仍只按 pooled final-sym full complex NRMSE、max-column、tie<=1e-6 时 earlier update 选 checkpoint。TRAIN diagonal、VAL gain、INTERNAL_TEST、burned24 不参与 selection。

R8E 有序树为 D→A→B→C→E：先 cancellation，再 readiness，再严格比较 H4g/H3g 和 H4m/H3m，相等不算改善。20 canonical +10000 random 共 10020 组：unclassified=0，multi=0，exactly-one=10020。

Dummy 实际模拟 32768 field +1024 diagonal pairs 的 forward/backward：field/diag/total loss=1.9960067/1.0326349/2.0992701，field/diag/combined grad norm=0.15500004/2.2802651/0.29790695，peak allocated/reserved=0.456559/0.558594 GiB，seconds/update=0.0265857，20k 纯计算估计=0.147698 h。未构造 optimizer，未 optimizer.step。

## Science isolation 和冻结说明

science optimizer constructed=false，science updates=0，H4 training=false，checkpoint inference=0，new exact physics/PSF/complete columns=0，fresh30=0/0，operator=0，Bunny/NVR/exact-F0=0。

本轮没有训练H4。

H4以H3为parent，
保持相同414210参数、165维输入、
boundary features、source masks、
dual-orientation field loss、
Hermitian inference、140/20/20 split、
field sample/C4 stream、optimizer和20k budget。

唯一新增science change是
TRAIN140-derived dense unit-diagonal consistency：

从每个update已有且blacklist-safe的field query coordinates
确定性选取1024个q，
构造(q,q,d=0)，
用TRAIN140 normalized diagonal median=1.0
作为approximate prior target，
增加：

`L_total = L_field + 0.1 L_diag`。

没有使用VAL、INTERNAL_TEST或burned24
定义该prior；
没有hard-code prediction center；
没有post-hoc column normalization；
没有lambda sweep。

没有生成新exact PSF，
没有densification，
没有fresh30，
没有operator，
没有Bunny/NVR/exact-F0。
