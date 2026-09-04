# P6 Phase-A8MRN2R8B Boundary and Gain Residual Mechanism Preregistration

## 结论

终态分类：`P6_PHASEA8MRN2R8B_BOUNDARY_AND_GAIN_RESIDUAL_MECHANISM_PREREGISTERED`；`R8C_execution_ready=true`。唯一 H3 candidate 为 `H2_BOUNDARY_AWARE_SOURCE_GEOMETRY_FIXED_FEATURES`。本轮只完成 R8 硬冻结、残差机制冻结、TRAIN140-only diagonal prior、H3 source wrapper 和 synthetic/dummy 审计；未训练 H3。

## R8 freeze 与 H2 baseline

R8 报告和 17 个 JSON（共 18 件）已逐一 SHA256 冻结并与 R8 提交清单核对，gate=PASS。R8 targeted-z outcome=`NO_GO`、`Z_SUPPORT_PRIMARY=false`、`R9_z_support_execution_ready=false`；z-midpoint `[15,45,74,104]` 与 144-column candidate 未重新激活，new support columns=0，new physics=0。

H2 best update=20000，VAL pooled/max=0.055862414242/0.0792927132051；INTERNAL full median/max=0.063451588638/0.0695806333254，`CANCEL2=false`。burned24 full median/max=0.0666122168046/0.112842946629，energy=0.0332643967766/0.0820776762631，boundary=0.0669623736417/0.115928036042，gain=0.0411827426218/0.148482587609，peak max=0。当前状态为 `NEAR_GATE_MEDIAN_WITH_RESIDUAL_OUTLIERS`。

## Boundary/corner 与 gain evidence

full 对 bx/by/bz/bmin 的 Pearson/Spearman 分别为 -0.619869288859/-0.698215202892、-0.455131755156/-0.46311235252、-0.310743019123/-0.135749628884、-0.396793677322/-0.429602992346。六 strata full median 为 `{'interior': 0.05624652105042159, 'x_boundary_near': 0.06783030697105288, 'y_boundary_near': 0.06394041130351664, 'z_low': 0.07161032874583839, 'z_high': 0.06140126017782724, 'corner_or_mixed_boundary': 0.09062327057886223}`；corner/interior ratio=1.6111800141。最坏 full 点 `A8M_FRESH_CORNER_OR_MIXED_BOUNDARY_02`，xyz=[147, 1, 5]，full/boundary/gain=0.112842946629/0.115928036042/0.122411350046。因此冻结 `BOUNDARY_CORNER_RESIDUAL_STRUCTURE_PRESENT=true`，但证据仅为 development descriptive evidence，不是 fresh gate。

gain max=0.148482587609，gate=0.02。gain 对 full/boundary/energy 的 Pearson/Spearman 为 0.6748163146/0.752173913043、0.598962174291/0.617391304348、0.136664929578/0.0982608695652。高 gain interior 点 `A8M_FRESH_INTERIOR_04` 的 gain=0.126759099686，因此冻结 gain residual 与 full/boundary 相关、但不只由 boundary 解释。

历史 burned24 exact diagonal normalized min/median/max/std=0.998423207373/1.00043859256/1.00354537149/0.00133217720194，imag max=0；H2 gain max 为其最大真实对角偏差的 41.8806852686 倍。未 hard-code center、未重归一化、未加入 diagonal/gain loss。

## TRAIN140-only diagonal prior

TRAIN140 exact diagonal normalized min/median/max/mean/std/span/max|deviation from 1|=0.997901328207/1/1.0039199728/1.00030414041/0.0013447644067/0.00601864459298/0.00391997279957，imag max=0。只使用已保存 TRAIN140 值；VAL20、INTERNAL_TEST20、burned24 均未参与 future training-prior 判断。该审计不改变本轮 H3 candidate，也不激活 gain loss。

## H3 unique source-representation change

H3 完整保留 H2 source x/y raw+k0..2、source z raw+k0..1，source z-k2 继续为零，source k4..7 继续为零；output raw+k0..7 与 relative raw+k0..9 完全不变。只把 H2 原本固定为零的六个 source k3 slots 改作 boundary auxiliary slots。

特征固定为 `[px,py,pz,pxy,pxz,pyz]`，其中 `px=abs(sx)`、`py=abs(sy)`、`pz=abs(sz)`，其余为对应乘积；完全由 normalized source coordinate 决定，不用 target、阈值、尺度或 stratum label。实际全局 tensor indices=[60, 76, 92, 68, 84, 100]。input=165，params=414210，model seed=20260909，fresh initial-state SHA=`7c4601d5c779655bc2a69482995868aa501310f5de3d1a12470708202424bb28`，禁止 H2 checkpoint warm-start。

`boundary_features_enabled=false` 时编码与 forward 均 bitwise 等于 H2，4096 pairs 最大归一化 discrepancy=0（门限 1e-7）。A branch 从 second=r 计算 feature，R branch 从 second=u 计算。100000 个随机 native source 的 R90 feature transform 最大误差=0（FP32 tolerance=1e-06）。H3 Hermitian discrepancy=0（门限 1e-6）；no detach 与有限梯度均 PASS。

Feature support 使用唯一坐标而非 occurrence frequency：TRAIN source branch 覆盖 C4-closed 140 anchors；reciprocal source-like branch 继承 R7 stream 对全部 2699864 个非黑名单 native 坐标的覆盖，x/y/z level=150/150/120。完整 sample stream 未保存。

## 为什么 H3 不加入 gain loss

同时加入 boundary geometry features 和 diagonal/gain loss 会使 H2→H3 改善来源不可辨识。因此 H3 只改变 source representation。仅当 H3 仍未 READY，才允许 R8D 另行预注册 TRAIN140-derived diagonal/gain consistency；VAL/TEST/burned 不得用于定义该 future prior。

## R8C training、selection 与 outcome

split={'TRAIN': 140, 'VAL': 20, 'INTERNAL_TEST': 20}，split SHA=`29900f4738d26f36893d3ebb5aae795b32d5836888a267c4bee98d440372a9ac`；blacklist unique=136，SHA=`0a733fd0328cd30377a4faf1a78432698da8a96f9c41476e6d233a39dc24754f`。每 update 4 sources、8192 queries/source、32768 pairs，shell=25/25/25/25%，model/sampler/C4 seed=20260909/20260916/20260913，sample-stream SHA=`6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2`。AdamW lr=2e-4、betas=(0.9,0.99)、weight decay=1e-6、scheduler=null、20k updates、VAL/checkpoint every500、no early stopping 均不变。

VAL 为 20 sources ×65536 probes，coordinate SHA=`718c4152461367368be80e02408c18667da09b18dc5550327337ccc06c8fb24b`；primary/secondary/tie 规则保持 pooled final-sym、max-column、earlier update。burned/gain/raw/strata 不参与 selection。dual loss 仍为 `0.5*MSE_complex(A,y)+0.5*MSE_complex(B,y)`，final=`0.5*(A+B)`，不增加任何 loss。

R8C ordered tree 冻结为 D→A→B→C→E；`H3m==H2m` 不算 full improvement，`H3g==H2g` 不算 gain improvement，无 epsilon、百分比或 p-value gate。20 canonical +10000 random 共10020组合：unclassified=0，multi=0，exactly-one=10020。

Dummy 32768 pairs 在 `NVIDIA GeForce RTX 4060 Laptop GPU` 上完成 dual forward/backward：peak allocated/reserved=0.452554/0.544922 GiB，seconds/update=0.135756，20k纯计算估计=0.754198 h；未构造 optimizer，未 optimizer.step。

## Science isolation

science optimizer constructed=false，science optimizer updates=0，H3 training=false，checkpoint inference=0，new exact physics/PSF/complete column=0，fresh30=0/0，operator=0，Bunny/NVR/exact-F0=0。

## 冻结说明

本轮没有训练H3。
R8已经正式NO-GO targeted-z densification，
因此没有新增任何PSF。

H3只在H2基础上增加一个source-boundary inductive bias：
利用原本固定为0的SOURCE-LIKE k=3六个输入slots，
放入由source coordinate确定的
px、py、pz、pxy、pxz、pyz。
没有恢复高频Fourier，
没有增加输入维数或参数量。

H3不加入center/gain loss，
因为boundary conditioning与gain consistency
必须分开进行controlled evaluation。
TRAIN140 diagonal只作为未来gain-mechanism的独立依据，
VAL/TEST/burned不会用于定义该future training prior。

没有checkpoint science inference，
没有新exact physics，
没有fresh30，
没有operator，
没有Bunny/NVR/F0。
