$script = "train-ablation.py"
$steps  = 1000000
$seeds  = 0,1,2

function RunConfig($exp, $extraArgs) {
  foreach ($s in $seeds) {
    Write-Host "===== RUN: $exp seed=$s ====="
    python $script --exp_name $exp --seed $s --total_timesteps $steps $extraArgs
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $exp seed=$s" }
  }
}

# Baseline
RunConfig "base" ""

# gSDE
RunConfig "sde" "--use_sde"

# KL-stop
RunConfig "kl" "--target_kl 0.03"

# RND
RunConfig "rnd" "--use_rnd --rnd_scale_start 0.05 --rnd_anneal_steps 250000"

# gSDE + KL
RunConfig "sde_kl" "--use_sde --target_kl 0.03"

# gSDE + RND
RunConfig "sde_rnd" "--use_sde --use_rnd --rnd_scale_start 0.05 --rnd_anneal_steps 250000"

# RND + KL
RunConfig "rnd_kl" "--use_rnd --rnd_scale_start 0.05 --rnd_anneal_steps 250000 --target_kl 0.03"

# gSDE + RND + KL (full stack)
RunConfig "sde_rnd_kl" "--use_sde --use_rnd --rnd_scale_start 0.05 --rnd_anneal_steps 250000 --target_kl 0.03"


Write-Host "ALL RUNS COMPLETE."
