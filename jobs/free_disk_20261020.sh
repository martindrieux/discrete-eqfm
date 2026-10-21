#!/bin/bash
# User-approved 2026-10-20 ("do the quick wins + ImageNet runs"). Frees space on /data/user_data/mdrieux.
# Quick wins: local copies of files on HF EquilibriumMap/eqfm-imagenet-distill (sha256 checked equal), a partial save,
# smoke-test checkpoints. Then the checkpoints of the ImageNet EqFM runs in eqfm-results (logs and results kept).
df -h /data/user_data/mdrieux | tail -1
# quick wins
rm -fv /data/user_data/mdrieux/eqfm-ckpts/justin_hf/cFAST_from85000/step_0120000.pt
rm -fv /data/user_data/mdrieux/eqfm-ckpts/justin_hf/ckpt/step_0085000.pt
rm -fv /data/user_data/mdrieux/eqfm-ckpts/eqmft_B30k_ema_merged.pt
rm -fv /data/user_data/mdrieux/deqfm/runs/TS_joint_lagrangian/last.pt.tmp
rm -fv /data/user_data/mdrieux/eqfm-results-smoke/smoke_wandb_pool_1907/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results-smoke/smoke_wandb_pool_1907/checkpoints/step_0000040.pt
# ImageNet EqFM run checkpoints
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_A_head/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_B_cosine/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_B_cosine/checkpoints/step_0040000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_B_full/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_B_full/checkpoints/step_0030000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_T_trunc08/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_T_trunc08/checkpoints/step_0018000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_TT_trunc08_teacher/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/eqmft_TT_trunc08_teacher/checkpoints/step_0003500.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/kept/step_0020000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/kept/step_0027300.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/kept/step_0030000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/kept/step_0037000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel/kept/step_0040000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel_resume40k/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel_resume40k/kept/step_0045000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel_resume40k/kept/step_0050000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel_resume40k/kept/step_0055000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2_lagsemi_cfg15_gnorm_babel_resume40k/kept/step_0060000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_justin60k_shiftmf_onpolK4_w0p3/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_justin60k_shiftmf_onpolK4_w0p3/checkpoints/step_0070000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_justin60k_shiftmf_onpolK4_w0p3/kept/step_0070000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k/checkpoints/latest.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k/kept/step_0010000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k/kept/step_0020000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k/kept/step_0030000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/s2mf_p1_semi_anchor_endpoint_cfg15_gnorm_from_lag27k/kept/step_0040000.pt
rm -fv /data/user_data/mdrieux/eqfm-results/stage1_eqm_uw_sitinit_babel/checkpoints/step_0004000.pt
df -h /data/user_data/mdrieux | tail -1
