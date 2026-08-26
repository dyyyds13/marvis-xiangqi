# Xiangqi RL training auto-start script.
# If a train_worker process already exists, do nothing; otherwise start training in background.
$found = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_worker' })
if ($found.Count -gt 0) {
    exit 0
}
# low-memory mode: workers=2 (fewer procs), batch=64 (smaller batch), sims=50
Start-Process "E:\Desktop\xiangqi\marvis\.venv\Scripts\python.exe" -ArgumentList @('-m','trainer.train_worker','--workers=2','--sims=50','--batch=64','--model-dir=models','--log-dir=logs') -WorkingDirectory "E:\Desktop\xiangqi\marvis" -WindowStyle Hidden
