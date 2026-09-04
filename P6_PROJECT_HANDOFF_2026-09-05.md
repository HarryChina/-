# P6_PROJECT_HANDOFF_2026-09-05

## 1. Project and environment
- Project root: `F:\psf模糊核`
- Original data/code: `D:\SAS_DATASET` (READ ONLY)
- GPU: RTX 4060 Laptop 8GB
- Python: `C:\Users\Harry\.conda\envs\pytorch\python.exe`
- GitHub repo for authoritative artifacts: `HarryChina/-`
- Prefer reading latest authoritative JSON/MD from GitHub connector instead of relying on pasted summaries.

## 2. Mandatory workflow rules
For every Codex round:
- If expected runtime >2 min, launch in background.
- Observe only 30–60 s, then return.
- Expose `status.json`, stdout, stderr.
- Report PID, exact CommandLine, phase/update, GPU/VRAM/RAM, log paths, monitor command.
- `Ctrl+C` stops monitor only.
- Explicit terminal state: `FINALIZED` / `FAILED`.
- Every round must state exactly which files to upload.

Submission folder is always:
`F:\psf模糊核\本轮提交`

Rules:
1. Clear contents at start, keep folder.
2. Finalizer COPYs required small files; never MOVE.
3. Create `本轮提交清单.json` and `.txt`.
4. Manifest fields: original_path, copied_path, bytes, SHA256, source_SHA256, hash_match, required_or_optional, purpose.
5. All hashes must match.
6. Prefer TOTAL<=20.
7. Never submit checkpoints, PSF/prediction arrays, large npy/npz, caches, stdout/stderr logs, large tensors.

## 3. Core science definitions
Formal branch: `P6_COMPLEX_PSF_FIELD_DECONV`

Normal operator:
`G=A0^H A0`

Exact physics PSF for source r:
`h_r = G δ_r = A0^H A0 δ_r`

Important wording:
- `A0 δ_r` is a synthetic SAS measurement signature, not a PSF.
- “real PSF” here means exact physics-computed reference PSF, not experimentally measured PSF.
- `A0^H` is adjoint/backprojection, not inverse.
- `G` is a matrix whose column is the position-dependent PSF.
- Current P6 PSF is in the linear normal/backprojection domain.
- Do not claim direct deconvolution of NVR output by G is justified.
- Valid high-level route: `raw y -> A0^H y ≈ Gx -> PSF deconv -> optional NVR prefit -> exact nonlinear F0 refinement`.

## 4. Frozen geometry / split / optimizer
Native volume:
`150×150×120 = 2,700,000`

Original exact anchor lattice:
- x/y `[0,30,60,89,119,149]`
- z `[0,30,60,89,119]`
- total 180 exact PSF columns

Frozen split:
- TRAIN 140
- VAL 20
- INTERNAL_TEST 20
- SHA `29900f4738d26f36893d3ebb5aae795b32d5836888a267c4bee98d440372a9ac`

C4 blacklist:
- unique 136
- SHA `0a733fd0328cd30377a4faf1a78432698da8a96f9c41476e6d233a39dc24754f`

VAL probe SHA:
`718c4152461367368be80e02408c18667da09b18dc5550327337ccc06c8fb24b`

`g_ref=25912.966796875`

Field sample-stream SHA:
`6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2`

Sampler:
- 4 sources/update
- 8192 queries/source
- 32768 field pairs/update
- 25/25/25/25 shell fractions
- model seed 20260909
- sampler seed 20260916
- C4 seed 20260913

Optimizer:
- AdamW lr=2e-4
- betas=(0.9,0.99)
- weight_decay=1e-6
- scheduler none
- exactly 20000 successful optimizer updates
- VAL/checkpoint every500
- no early stopping

Checkpoint selection:
1. lowest pooled final-sym full complex NRMSE
2. then lowest max-column if primary diff<=1e-6
3. then earlier update
Never use raw/gain/TEST/burned/fresh for selection.

## 5. Development chronology
### D0
Direct INR, params=414210.
Same-lattice INTERNAL looked ~5%, but off-lattice burned24 full median:
`0.5347725653959421`

### R1
Discovered hidden Hermitian branch cancellation:
two raw branches ~5.26 error cancelling to final ~0.053.

### H0 / R3
Dual-orientation loss:
`0.5*MSE(A,y)+0.5*MSE(B,y)`
Cancellation repaired.
burned24 full median:
`0.390950109101022`

### H1 / R5
Only source absolute Fourier changed:
- keep raw xyz
- keep source k0..2
- zero source k3..7
- output k0..7 unchanged
- relative d k0..9 unchanged
Result burned24 full median:
`0.15019320777511364`

### H2 / R7
Anisotropic source smoothing:
- source x/y keep k0..2
- source z keep k0..1
- source z-k2 zero
Result:
- full median/max `0.06661221680459492 / 0.11284294662938457`
- energy `0.03326439677655908 / 0.08207767626306936`
- boundary `0.06696237364173963 / 0.11592803604241879`
- gain `0.04118274262184512 / 0.14848258760888638`
- peak max 0
- cancellation false
Targeted-z densification later became NO-GO.

### H3 / R8C
Added boundary geometry using six previously-zero source k3 slots:
`px,py,pz,pxy,pxz,pyz`
slots `[60,76,92,68,84,100]`

H3 result:
- best update 18000
- best VAL pooled/max `0.05320869817865062 / 0.07868135918409411`
- INTERNAL full `0.058477638378815554 / 0.07084092214014827`
- CANCEL3=false
- burned full `0.06660179578093173 / 0.12139477236343238`
- energy `0.01788038221192561 / 0.05499024150929448`
- boundary `0.06634186330782878 / 0.12631352870541893`
- gain `0.07052240806570562 / 0.18721155057525807`
Interpretation:
- full median essentially unchanged
- energy improved strongly
- full max/corner/gain worsened
- do not continue boundary-feature sweep

## 6. burned24 vs fresh30
burned24:
- 24 off-lattice source positions
- each prediction is a full 150×150×120 complex PSF
- repeatedly inspected during development
- therefore now a development diagnostic set

fresh30:
- untouched off-lattice source set
- never selected/accessed yet
- used only after final development freeze
- purpose: test whether success generalizes beyond burned24

Do not claim whole-domain formal success until fresh30.

## 7. R8D H4 preregistration
Completed stage:
`P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTRATION`

Classification:
`P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTERED`

`R8E_execution_ready=true`

H4 candidate:
`H4_H3_TRAIN140_DENSE_UNIT_DIAGONAL_CONSISTENCY_LAMBDA0P1`

Parent: H3
Architecture exactly H3:
- params 414210
- input 165
- boundary features unchanged
- source masks unchanged
- output k0..7 unchanged
- relative k0..9 unchanged
- inference `0.5*(A+B)`
- no hard-coded center / renorm / clipping

H4 helper:
`F:\psf模糊核\scripts\p6_phaseA8MRN2R8D_diagonal_gain_consistency_model.py`

SHA:
`a05996652c19b30d084d92b57942fdbbfe44abef93987a42f8517346dba79c53`

TRAIN140 normalized diagonal:
- n 140
- min 0.9979013282065966
- median 1.0
- max 1.0039199727995733
- mean 1.0003041404061261
- std 0.001344764406703845
- max deviation from1 0.003919972799573346
- imag max 0

Prior isolation:
- target source = TRAIN140 median only
- VAL/INTERNAL_TEST/burned not used
- new physics = 0

Diagonal q:
- each update select 1024 q from existing post-C4 blacklist-safe field queries
- per source use fixed local indices `0,32,64,...,8160`
- construct `(q,q,d=0)`
- target `1+0j`
- no new RNG
- no target-error q selection

Diagonal stream SHA:
`395bd3a0e85405db340aab23a0fce4f334b95bf6750b029bbb307b399db526ce`

20k coordinate replay:
- 20,480,000 diagonal q occurrences
- 2,698,300 unique native coords
- 150/150/120 levels
- blacklist hits 0

Loss:
`L_field = 0.5*MSE(A,y)+0.5*MSE(B,y)`
`L_diag = 0.5*MSE(A_diag,1)+0.5*MSE(B_diag,1)`
`lambda_diag=0.1`
`L_total=L_field+0.1*L_diag`

Important dummy gradient context:
- field grad norm ~0.15500004
- raw diag grad norm ~2.2802651
- weighted diag grad ~0.2280265
- weighted diag / field ~1.47
Do not change lambda post hoc.

R8E outcome tree frozen:
`D -> A -> B -> C -> E`
- D cancellation
- A all readiness pass
- B gain and full improve, not ready
- C gain improves, full does not
- E otherwise
No epsilon.

## 8. Current operational incident: R8E failed mid-run
R8E H4 formal worker:
`F:\psf模糊核\scripts\p6_phaseA8MRN2R8E_H4_development.py`

Original command:
`"C:\Users\Harry\.conda\envs\pytorch\python.exe" "F:\psf模糊核\scripts\p6_phaseA8MRN2R8E_H4_development.py" --run --device cuda`

Original PID 50364, now exited.

Failure:
`Nonfinite H4 gradient at update 18066`

Latest complete checkpoint:
`F:\psf模糊核\results\airsas_bunny20k\p6_complex_psf_field_deconv\phaseA8MRN2R8E_work\checkpoints\H4_update_18000.pt`

Checkpoint readable and lineage matched.

Best before failure:
- update ~17000
- VAL pooled/max ~`0.0840415 / 0.1292353`
(use authoritative status/history for exact values)

No leakage:
- INTERNAL_TEST unopened
- burned24 unopened
- fresh30=0/0
- operator=0
- blacklist hits 0

Probable root cause:
- GradScaler scale at update18000 grew `16,777,216 -> 33,554,432`
- later scaled FP16 gradient overflow
- worker treated nonfinite gradient as fatal before normal GradScaler skip/downscale recovery
- likely implementation incident, not science failure

Do NOT directly use old `--resume`.
Do NOT manually lower checkpoint scaler before replay.

## 9. Current next stage
Next required stage:
`P6_PHASEA8MRN2R8EA_AMP_GRADSCALER_OVERFLOW_RECOVERY_AMENDMENT`

This is implementation-only, not new science.

Required semantics:
- old worker remains read-only
- restore checkpoint18000 exactly, including scaler/RNG
- replay 18001 onward
- at an overflow:
  - do not count attempt as successful update
  - GradScaler skips optimizer step and reduces scale
  - verify parameters and optimizer counters unchanged
  - retry the SAME logical batch with reduced scale
  - only successful optimizer.step increments logical update
- do not consume new sample on failed AMP attempt
- stream hash appends once per logical update, not per retry
- if underlying loss itself is nonfinite, stop as real numerical failure
- safe retry cap may be 8
- science diff count must be 0

Suggested patched worker:
`F:\psf模糊核\scripts\p6_phaseA8MRN2R8EA_H4_amp_recovery_resume.py`

Suggested resume command:
`"C:\Users\Harry\.conda\envs\pytorch\python.exe" "F:\psf模糊核\scripts\p6_phaseA8MRN2R8EA_H4_amp_recovery_resume.py" --resume-from "F:\psf模糊核\results\airsas_bunny20k\p6_complex_psf_field_deconv\phaseA8MRN2R8E_work\checkpoints\H4_update_18000.pt" --device cuda`

R8EA must not modify:
- model
- parent
- boundary features
- source masks
- lambda=0.1
- losses
- q selection
- field/diag streams
- optimizer hyperparameters
- 20000 successful update budget
- VAL selection
- TEST/burned protocol
- R8E outcome tree

Do not add:
- gradient clipping
- lr change
- lambda change
- manual scaler pre-reset
- center hard-code
- post-hoc normalization
- second H4 lineage
- restart-from-zero and select best run

After R8EA PASS:
resume H4 in background automatically, no extra user confirmation.

## 10. Frozen readiness gates
- full median <=0.06
- full max <=0.10
- energy median <=0.04
- energy max <=0.06
- boundary median <=0.10
- boundary max <=0.16
- peak max <=1
- center gain max <=0.02

Cancellation gate:
CANCEL iff all:
1. INTERNAL final median<=0.08
2. raw_mean/final>=1.5
3. median error cosine<=-0.25

## 11. Sealed future data/work
Until final development pass and explicit prereg:
- fresh30 selected=0
- fresh30 accessed=0
- operator cases=0
- Bunny deconvolution=0
- NVR updates=0
- exact-F0=0
- no new exact PSF generation

## 12. New-window instructions
The new ChatGPT window must first:
1. Read this handoff.
2. Use GitHub connector to inspect latest authoritative files in `HarryChina/-`.
3. Check whether R8EA has already been executed after this file was written.
4. If not, continue with implementation-only R8EA.
5. If yes, read R8EA report/status/events/resume-lineage audit and determine whether H4 resumed/finalized.
6. Never invent or alter frozen science protocol without a new explicit prereg stage.

## 13. First message to paste into the new window
我正在继续一个长期的 P6 complex PSF-field INR / SAS 项目。请先完整读取我提供的 `P6_PROJECT_HANDOFF_2026-09-05.md`，并使用已连接的 GitHub 连接器读取仓库 `HarryChina/-` 中最新的 R8D、R8E、R8EA（如果已存在）正式工件。不要仅依赖聊天摘要，也不要自行改变冻结协议。

当前已知最后状态是：R8D 已完成 H4 diagonal/gain consistency 预注册；R8E 的 H4 正式训练在 update 18066 因 AMP/GradScaler overflow handling 的实现问题中断，最新完整 checkpoint 是 update 18000。下一步原则上是 implementation-only 的 `P6_PHASEA8MRN2R8EA_AMP_GRADSCALER_OVERFLOW_RECOVERY_AMENDMENT`，从 checkpoint18000 严格 same-lineage、same-batch retry 恢复，science diff 必须为0；但请先从 GitHub 核实是否已经有更新的 R8EA/R8E 终态文件。

之后继续沿用旧窗口规则：每轮 Codex 如果预计>2分钟必须后台；只观察30–60秒；必须有 status.json、stdout、stderr、PID、完整 CommandLine 和 monitor；最终状态明确 FINALIZED/FAILED；每轮开始清空 `F:\psf模糊核\本轮提交` 内容但不删目录，finalizer 用 COPY 而不是 MOVE，生成 `本轮提交清单.json/.txt`，源/副本 SHA256 必须一致，TOTAL 尽量<=20，不放 checkpoint、PSF/prediction arrays、large npy/npz、logs。每次给我下一步 Codex 指令时，都必须在末尾明确本轮需要提交/上传哪些文件。

先告诉我：你从 handoff + GitHub 中确认到的“当前正式阶段、最新终态/中断状态、下一步唯一允许动作”分别是什么，然后再给出下一步指令。
