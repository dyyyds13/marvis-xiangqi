# -*- coding: utf-8 -*-
"""多进程训练器冒烟测试：2 workers 跑 30 秒"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.train_worker import Trainer

if __name__ == '__main__':
    trainer = Trainer(
        model_dir='models_test', log_dir='logs_test',
        num_workers=2, num_simulations=30, batch_size=64,
        train_every=16, save_every=5, eval_every=10, eval_games=2,
    )
    trainer.start_workers()
    t0 = time.time()
    try:
        while time.time() - t0 < 30:
            added = trainer._drain_queue()
            if len(trainer.buffer) >= trainer.batch_size and added > 0:
                from trainer.train import train_step
                for _ in range(trainer.train_every):
                    if len(trainer.buffer) < trainer.batch_size:
                        break
                    batch = trainer.buffer.sample(trainer.batch_size)
                    stats = train_step(trainer.net, batch, trainer.optimizer)
                    trainer.train_steps += 1
                if trainer.train_steps % trainer.save_every == 0:
                    from trainer.train import save_model
                    save_model(trainer.net, trainer.model_path)
            time.sleep(0.2)
    finally:
        trainer.stop()
    print(f'\n=== 冒烟测试结果 ===')
    print(f'训练步数: {trainer.train_steps}')
    print(f'对局数: {trainer.total_games}')
    print(f'样本数: {trainer.total_samples}')
    print(f'回放池大小: {len(trainer.buffer)}')
    print(f'模型文件存在: {os.path.exists(trainer.model_path)}')
    print('=== 测试完成 ===')
