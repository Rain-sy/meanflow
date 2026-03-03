import os
import random
import glob
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 🌟 必须和原本 build_dataset 的种子保持一致！
random.seed(42)

def add_ffhq_incremental(ffhq_source, target_hr_dir, target_lr_dir, 
                         prev_count=3550, add_count=1000):
    """
    ffhq_source: FFHQ 原始大图目录
    target_hr_dir: 你现在的 Mix15K_HR 目录
    prev_count: 之前已经选了多少张 (3550)
    add_count: 这次要加多少张 (1000)
    """
    
    # 1. 获取所有 FFHQ 原图列表
    valid_exts = {'.png', '.jpg', '.jpeg'}
    all_files = sorted([f for f in os.listdir(ffhq_source) if Path(f).suffix.lower() in valid_exts])
    
    if len(all_files) < (prev_count + add_count):
        print(f"❌ FFHQ 总图数 ({len(all_files)}) 不够！需要 {prev_count + add_count} 张。")
        return

    # 2. 核心逻辑：重演之前的随机抽样
    # Python 的 random.sample 在种子固定时，顺序是确定的。
    # 我们先抽取前 3550 张（这些是你已经有的，我们要避开）
    print(f"🔍 正在回溯之前的随机选择 (Seed=42)...")
    
    # 注意：为了避开已选的，最稳妥的方法是先 shuffle 全部，然后切片
    # 假设之前的逻辑是 random.sample(all, 3550)
    # random.sample 的内部机制比较复杂，为了确保绝对的互斥，我们使用更稳妥的集合排除法
    
    # 重新模拟之前的选择
    already_selected = set(random.sample(all_files, prev_count))
    
    # 找出剩下的
    remaining_files = [f for f in all_files if f not in already_selected]
    
    print(f"   FFHQ 总数: {len(all_files)}")
    print(f"   已用: {len(already_selected)}")
    print(f"   剩余可用: {len(remaining_files)}")
    
    # 3. 从剩余的中再选 1000 张
    new_selection = random.sample(remaining_files, add_count)
    
    print(f"🚀 准备追加 {len(new_selection)} 张新图片到现有数据集...")
    
    # 4. 开始处理并追加
    # 文件名接着之前的序号：FFHQ_03550.png, FFHQ_03551.png ...
    start_index = prev_count 
    
    for i, filename in enumerate(tqdm(new_selection, desc="追加 FFHQ")):
        try:
            img_path = os.path.join(ffhq_source, filename)
            hr_img = Image.open(img_path).convert('RGB')
            
            # 这里的序号 i 从 0 开始，所以文件名序号 = start_index + i
            # 例如：第 1 张新图 -> FFHQ_03550.png
            current_idx = start_index + i
            new_filename = f"FFHQ_{current_idx:05d}.png"
            
            hr_save_path = os.path.join(target_hr_dir, new_filename)
            lr_save_path = os.path.join(target_lr_dir, new_filename)
            
            # 1. 保存 HR
            hr_img.save(hr_save_path, format='PNG')
            
            # 2. 生成 LR (Bicubic x4)
            W, H = hr_img.size
            lr_img = hr_img.resize((W // 4, H // 4), Image.BICUBIC)
            lr_img.save(lr_save_path, format='PNG')
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n✅ 追加完成！")
    print(f"   HR 目录总数应该是: {len(os.listdir(target_hr_dir))}")

if __name__ == "__main__":
    # 请修改为你的实际路径
    add_ffhq_incremental(
        ffhq_source="./Data/FFHQ",
        target_hr_dir="./Data/Mix15K_HR",
        target_lr_dir="./Data/Mix15K_LR_bicubic_X4",
        prev_count=3550,  # 之前选了多少
        add_count=1000    # 这次加多少
    )