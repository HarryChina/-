# P6 Phase-A8MRN2R8C Boundary-Aware Source Conditioning Development

**Classification:** `P6_PHASEA8MRN2R8C_BOUNDARY_AWARE_SOURCE_CONDITIONING_IMPROVES_H2_BUT_NOT_READY`  
**R8C outcome:** CASE B  
**Recommended next:** `P6_PHASEA8MRN2R8D_TRAIN_ONLY_DIAGONAL_GAIN_CONSISTENCY_PREREGISTRATION`

## Frozen controlled experiment

H3 completed exactly 20000 optimizer updates with 414210 parameters, 165-dimensional input, fresh initialization, dual-orientation supervision, Hermitian final inference, frozen 140/20/20 split, sample/C4 stream, optimizer, and budget. Relative to H2, its only science change was placing px, py, pz, pxy, pxz, and pyz in global input slots [60, 76, 92, 68, 84, 100]. SOURCE-LIKE z-k2 stayed zero; k4..7 stayed zero; OUTPUT-LIKE k0..7 and relative k0..9 were unchanged. No gain or other auxiliary loss was used. Sample-stream SHA `6639bcac3bb9c5efce5d43c5fb5e784e8f8a67a4c99cc75559c967369e57a7e2` matched.

Best update 18000; best pooled/max VAL 0.05320869818/0.07868135918; final pooled/max VAL 0.05330769897/0.09038880776.

## INTERNAL_TEST and cancellation retest

Full median/max 0.05847763838/0.07084092214; energy median/max 0.01757892458/0.04245915389; boundary median/max 0.0586053255/0.07040646617; gain max 0.03592560278; peak max 0.

Raw A/B/disagreement medians 0.06150276514/0.06101689195/0.03665391573; error cosine 0.807380883; cancellation ratio 0.9506500391. CANCEL3=`False`.

## burned24 and controlled mechanism comparisons

Full median/max 0.06660179578/0.1213947724; energy median/max 0.01788038221/0.05499024151; boundary median/max 0.06634186331/0.1263135287; gain median/max 0.07052240807/0.1872115506; peak max 0.

- corner_or_mixed_boundary: full median 0.099890415
- interior: full median 0.058635654
- x_boundary_near: full median 0.070074456
- y_boundary_near: full median 0.064714676
- z_high: full median 0.057057209
- z_low: full median 0.072568205

H3 vs H2 full/boundary/gain-max improvement = 0.0156443%/0.926655%/-26.0832%. Interior/corner/ratio = 0.05863565445/0.09989041504/1.70357807. READY3=`False`.

## Interpretation and isolation

R8C tests one preregistered boundary-geometry representation change only. No post-hoc feature search, threshold, added scale, corner indicator, second candidate, gain loss, or burned-driven retraining occurred. TRAIN140 diagonal values remained future context only.

The H3 best freeze preceded INTERNAL_TEST; cancellation retest completed before burned24 first access. No new exact PSF, complete PSF column, anchor densification, fresh30 selection/access, operator case, Bunny deconvolution, NVR, or exact-F0 was executed.
