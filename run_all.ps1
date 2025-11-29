# Run all experiments
$ErrorActionPreference = "Continue"

Write-Host "`n========================================"
Write-Host "Starting All Experiments"
Write-Host "========================================`n"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "Timestamp: $timestamp`n"

# Create directories
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "results" | Out-Null

$completedCount = 0
$failedCount = 0

# Experiment 1: PPO
Write-Host "`n[1/4] Running PPO (50000 steps)..." -ForegroundColor Cyan
D:\IsaacLab\isaaclab.bat -p algorithms\ppo.py --num-envs 128 --total-timesteps 50000 --headless 2>&1 | Tee-Object -FilePath "logs\ppo_$timestamp.log"
if ($LASTEXITCODE -eq 0) { $completedCount++ } else { $failedCount++ }
Write-Host "PPO Done!`n" -ForegroundColor Green

# Experiment 2: DDPG  
Write-Host "`n[2/4] Running DDPG (10000 steps)..." -ForegroundColor Cyan
D:\IsaacLab\isaaclab.bat -p algorithms\ddpg.py --num-envs 128 --total-timesteps 10000 --headless 2>&1 | Tee-Object -FilePath "logs\ddpg_$timestamp.log"
if ($LASTEXITCODE -eq 0) { $completedCount++ } else { $failedCount++ }
Write-Host "DDPG Done!`n" -ForegroundColor Green

# Experiment 3: TD3
Write-Host "`n[3/4] Running TD3 (50000 steps)..." -ForegroundColor Cyan
D:\IsaacLab\isaaclab.bat -p algorithms\td3.py --num-envs 128 --total-timesteps 50000 --headless 2>&1 | Tee-Object -FilePath "logs\td3_$timestamp.log"
if ($LASTEXITCODE -eq 0) { $completedCount++ } else { $failedCount++ }
Write-Host "TD3 Done!`n" -ForegroundColor Green

# Experiment 4: SAC
Write-Host "`n[4/4] Running SAC (10000 steps)..." -ForegroundColor Cyan
D:\IsaacLab\isaaclab.bat -p algorithms\sac.py --num-envs 128 --total-timesteps 10000 --headless 2>&1 | Tee-Object -FilePath "logs\sac_$timestamp.log"
if ($LASTEXITCODE -eq 0) { $completedCount++ } else { $failedCount++ }
Write-Host "SAC Done!`n" -ForegroundColor Green

# Summary
Write-Host "`n========================================"
Write-Host "All Experiments Complete!"
Write-Host "========================================"
Write-Host "Completed: $completedCount / 4"
Write-Host "Failed: $failedCount / 4"
Write-Host "`nLogs: logs\"
Write-Host "Results: results\`n"
