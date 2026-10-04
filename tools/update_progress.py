# -*- coding: utf-8 -*-
"""
象棋 RL 训练进度自动更新脚本
用法: .venv/Scripts/python.exe tools/update_progress.py
功能: 解析 logs/train.log -> 估算累计训练步数 -> 生成进度图 PNG + HTML 页面
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
import html
import sys
from datetime import datetime

# ---------- 路径 ----------
BASE = r"E:\Desktop\xiangqi\marvis"
LOG = os.path.join(BASE, "logs", "train.log")
OUT_DIR = r"C:\Users\HP\AppData\Roaming\Tencent\Marvis\User\oAN1i2fq-Lg-AhOSuiyBq3Ypsssw\workspace\conv_19ffee01457_f815d093a17c\output"
PNG = os.path.join(OUT_DIR, "train_progress.png")
HTML = os.path.join(OUT_DIR, "train_progress.html")

# ---------- 里程碑目标 ----------
MILESTONE_CLASSIC = 500_000   # 打败浅层经典 Alpha-Beta (经验区间 20~50 万步)
MILESTONE_USER = 3_000_000    # 打败业余中上(用户) (经验区间 100~300 万步)

# ---------- 解析日志估算累计步数 ----------
def calc_total_steps(log_path):
    """日志每会话从 0 计数，按会话增量求和估算累计步数"""
    sessions = []
    cur = None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.search(r'step=(\d+)', line)
                if m:
                    s = int(m.group(1))
                    if cur is None:
                        cur = {'start': s, 'max': s}
                    else:
                        cur['max'] = max(cur['max'], s)
                elif re.search(r'续训|加载已有模型|训练开始|启动 \d+ 个自对弈', line):
                    if cur is not None:
                        sessions.append(cur)
                        cur = None
        if cur is not None:
            sessions.append(cur)
    except FileNotFoundError:
        return 0, 0
    total = sum(max(0, s['max'] - s['start'] + 20) for s in sessions)
    return total, len(sessions)

total, session_cnt = calc_total_steps(LOG)

# ---------- 中文字体 ----------
font_path = r"C:\Windows\Fonts\msyh.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')

p1 = min(total / MILESTONE_CLASSIC, 1.0)
p2 = min(total / MILESTONE_USER, 1.0)

ax.text(0, 2.9, "里程碑 1：打败浅层经典 Alpha-Beta", fontsize=13, color='#cdd6f4', fontweight='bold')
ax.barh(2.5, MILESTONE_CLASSIC/10000, height=0.55, color='#313244', edgecolor='#45475a')
ax.barh(2.5, total/10000, height=0.55, color='#89b4fa')
ax.text(MILESTONE_CLASSIC/10000 + 1.5, 2.5, f"{p1*100:.0f}%", va='center', fontsize=13, color='#89b4fa', fontweight='bold')

ax.text(0, 1.9, "里程碑 2：打败你（业余中上棋力）", fontsize=13, color='#f5e0dc', fontweight='bold')
ax.barh(1.5, MILESTONE_USER/10000, height=0.55, color='#313244', edgecolor='#45475a')
ax.barh(1.5, total/10000, height=0.55, color='#f38ba8')
ax.text(MILESTONE_USER/10000 + 1.5, 1.5, f"{p2*100:.1f}%", va='center', fontsize=13, color='#f38ba8', fontweight='bold')

ax.set_xlim(0, 320)
ax.set_yticks([])
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xticklabels(['0', '50万', '100万', '150万', '200万', '250万', '300万'], color='#a6adc8', fontsize=11)
for spine in ax.spines.values():
    spine.set_color('#45475a')

left_classic = max(MILESTONE_CLASSIC - total, 0)
left_user = max(MILESTONE_USER - total, 0)
info = (f"当前累计训练：约 {total/10000:.1f} 万步（{session_cnt} 个训练会话）\n"
        f"按当前低内存速度（每日 3~6 万步）估算：\n"
        f"· 离打败浅层经典 AI 还差约 {left_classic/10000:.0f} 万步（约 {left_classic/40000:.0f}~{left_classic/30000:.0f} 天）\n"
        f"· 离打败你（业余中上）还差约 {left_user/10000:.0f} 万步（约 {left_user/60000:.0f}~{left_user/30000:.0f} 周）")
ax.set_title("象棋 RL 训练棋力进度", color='#cdd6f4', fontsize=18, fontweight='bold', pad=18)
ax.text(0, 3.5, info, fontsize=11, color='#a6adc8', va='top', linespacing=1.8)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(PNG, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)

# ---------- 生成 HTML 页面 ----------
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>象棋 RL 训练进度</title>
<style>
  body {{ background:#1e1e2e; color:#cdd6f4; font-family:'Microsoft YaHei',sans-serif; display:flex; flex-direction:column; align-items:center; padding:40px 20px; }}
  h1 {{ color:#f5e0dc; }}
  .card {{ background:#313244; border-radius:12px; padding:24px 32px; margin:16px 0; width:min(860px,100%); box-shadow:0 4px 20px rgba(0,0,0,.4); }}
  .stat {{ display:flex; justify-content:space-around; flex-wrap:wrap; gap:12px; }}
  .stat div {{ text-align:center; }}
  .num {{ font-size:28px; font-weight:bold; color:#89b4fa; }}
  .num2 {{ color:#f38ba8; }}
  .label {{ font-size:13px; color:#a6adc8; margin-top:4px; }}
  img {{ width:100%; border-radius:8px; }}
  .foot {{ color:#6c7086; font-size:12px; margin-top:12px; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:12px; background:#45475a; color:#cdd6f4; }}
</style>
</head>
<body>
<h1>♟ 象棋 RL 训练进度</h1>
<div class="card">
  <div class="stat">
    <div><div class="num">{total/10000:.1f} 万</div><div class="label">累计训练步数</div></div>
    <div><div class="num">{p1*100:.0f}%</div><div class="label">击败经典 AI 进度</div></div>
    <div><div class="num num2">{p2*100:.1f}%</div><div class="label">击败你（业余中上）进度</div></div>
  </div>
</div>
<div class="card"><img src="train_progress.png" alt="训练进度图"></div>
<div class="card">
  <div class="label">最近更新：{now}</div>
  <div class="label" style="margin-top:6px">
    里程碑估算为经验参考（经典 AI 20~50 万步，业余中上 100~300 万步），实际以对战胜率为准。
  </div>
</div>
<div class="foot">由 Marvis 自动更新 · 训练中每步自动落盘 current.pt</div>
</body>
</html>"""
with open(HTML, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"OK total={total} sessions={session_cnt}")
print("PNG:", PNG)
print("HTML:", HTML)

# ---------- 自动更新模式：训练运行期间定期刷新 ----------
if '--watch' in sys.argv:
    import subprocess
    import time

    def train_running():
        """检测 train_worker 训练进程是否在运行"""
        ps = (
            "Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
            "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_worker' }"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())

    print("watch mode: 每 10 分钟自动刷新进度页，训练结束后自动退出")
    while train_running():
        time.sleep(600)
        subprocess.run([sys.executable, os.path.abspath(__file__)],
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    print("watch: 训练进程已结束，退出")
