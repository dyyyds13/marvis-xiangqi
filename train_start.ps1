# Xiangqi RL training auto-start script.
# If a train_worker process already exists, do nothing; otherwise start training in background.
$found = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_worker' })
if ($found.Count -gt 0) {
    exit 0
}
# max-memory mode: workers=8 (CPU-rich, i7-13700H 20 threads), batch=128 (memory-heavy), sims=50
# Memory ~1.9GB total for training, leaves ~5GB free for daily work.
# Progress page (PNG + HTML) auto-refreshes inside train_worker every progress-every steps.
Start-Process "E:\Desktop\xiangqi\marvis\.venv\Scripts\python.exe" -ArgumentList @('-m','trainer.train_worker','--workers=8','--sims=50','--batch=128','--progress-every=100','--model-dir=models','--log-dir=logs') -WorkingDirectory "E:\Desktop\xiangqi\marvis" -WindowStyle Hidden


