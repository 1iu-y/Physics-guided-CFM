import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from matplotlib.lines import Line2D
import sys
src_path = ".../src"
if src_path not in sys.path:
    sys.path.append(src_path)
from twophasefvm import twophase_impes
from matplotlib.ticker import ScalarFormatter
import argparse

# ================= 配置区域 =================
FLOW_GT_FILE = '.../raw_data/q_total.txt'
PERM_FILE_FOR_INDEX = '.../raw_data/perms3d_trans.txt'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


# ===========================================

def extract_sample_id(filename):
    match = re.search(r'sample_(\d+).txt', filename)
    return int(match.group(1)) if match else -1


def main():

    parser = argparse.ArgumentParser(description="Plotting Saved QWT Results")

    parser.add_argument("--pred_data_dir", type=str, default= '.../production_file_name',
                        help="Directory containing predicted flow rates (from Step 2)")
    
    parser.add_argument("--obs_file", type=str, default='.../raw_data/subqwt_last200_noisy_6.5percent.txt',
                        help="Path to observation data file")
    
    parser.add_argument("--output_img_dir", type=str, default='.../vis_file_name',
                        help="Directory to save final plots")
    
    parser.add_argument("--perm_samples_dir", type=str, default='.../perms_file_name',
                        help="Directory containing original permeability samples (from Step 1)")

    args = parser.parse_args()

    # 将参数赋值给变量
    PRED_DATA_DIR = args.pred_data_dir
    OBS_FILE = args.obs_file
    OUTPUT_IMG_DIR = args.output_img_dir
    PERM_SAMPLES_DIR = args.perm_samples_dir

    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 1. 计算索引
    print("Calculating indices...")
    if not os.path.exists(PERM_FILE_FOR_INDEX):
        print(f"Error: {PERM_FILE_FOR_INDEX} not found.")
        return
    with open(PERM_FILE_FOR_INDEX, 'r') as f:
        N_total = sum(1 for _ in f)

    n_train = int(N_total * TRAIN_RATIO)
    n_val = int(N_total * VAL_RATIO)
    test_start_index = n_train + n_val
    print(f"Test set starts at global index: {test_start_index}")

    # 2. 加载数据
    print("Loading GT and Obs...")
    all_flow_gt = np.loadtxt(FLOW_GT_FILE)
    all_obs = np.loadtxt(OBS_FILE)

    # 3. 查找并排序文件
    pred_files = glob.glob(os.path.join(PRED_DATA_DIR, "q_pred_sample_*.txt"))
    post_files = glob.glob(os.path.join(PERM_SAMPLES_DIR, "sample_*.txt"))

    pred_files.sort(key=extract_sample_id)
    post_files.sort(key=extract_sample_id)

    if not pred_files:
        print(f"No data files found in {PRED_DATA_DIR}")
        return

    if len(pred_files) != len(post_files):
        print(f"Warning: File count mismatch. Pred: {len(pred_files)}, Post: {len(post_files)}")
        min_len = min(len(pred_files), len(post_files))
        pred_files = pred_files[:min_len]
        post_files = post_files[:min_len]

    print(f"Found {len(pred_files)} matched file pairs. Starting plotting...")

    # 4. 循环处理
    for pred_file, post_file in zip(pred_files, post_files):
        sample_idx = extract_sample_id(pred_file)

        # 二次确认 ID 是否匹配
        post_idx = extract_sample_id(post_file)
        if sample_idx != post_idx:
            print(f"Error: Mismatch ID between {pred_file} and {post_file}. Skipping.")
            continue

        print(f"Processing Sample {sample_idx}...")

        try:
            # 读取生成样本的预测流量
            q_pred_flat = np.loadtxt(pred_file)

            # 读取后验渗透率样本，计算均值，运行模拟
            all_perms = np.loadtxt(post_file)
            perms_mean = np.mean(all_perms, axis=0)
            sim_results = twophase_impes(2 ** perms_mean * 1e-16)

            if isinstance(sim_results, tuple):
                q_mean_raw = sim_results[0]
            else:
                q_mean_raw = sim_results

            q_mean = q_mean_raw.reshape(4, 1080)

        except Exception as e:
            print(f"  Error reading processing {pred_file}: {e}")
            continue

        if q_pred_flat.ndim == 1:
            q_pred_flat = q_pred_flat[np.newaxis, :]
        n_times = q_pred_flat.shape[0]
        q_gen_batch = q_pred_flat.reshape(n_times, 4, 1080)

        global_idx = test_start_index + sample_idx
        if global_idx >= all_flow_gt.shape[0]: continue
        q_true = all_flow_gt[global_idx].reshape(4, 1080)

        if sample_idx >= all_obs.shape[0]: continue
        q_obs = all_obs[sample_idx].reshape(4, 54)


        q_true_flat = q_true.reshape(-1)

        sigma_std_well =  np.abs(q_true[:, 0])      # (4,)
        sigma_std_full = np.repeat(sigma_std_well, 1080)       # (4320,)
        sigma_inv_full = 1.0 / (sigma_std_full ** 2)           # (4320,)

        def sigma_inv_norm(diff_flat, sigma_inv_flat):
            return np.sqrt(np.sum(diff_flat**2 * sigma_inv_flat))

        q_mean_flat = q_mean.reshape(-1)
        diff_mean = q_mean_flat - q_true_flat

        norm_true_sigma = sigma_inv_norm(q_true_flat, sigma_inv_full)
        norm_diff_mean = sigma_inv_norm(diff_mean, sigma_inv_full)
        rel_err_mean_field = norm_diff_mean / norm_true_sigma

        q_gen_flat = q_gen_batch.reshape(n_times, -1)          # (N,4320)
        diff_gen = q_gen_flat - q_true_flat[None, :]           # (N,4320)

        norm_diff_gen = np.sqrt(
            np.sum(diff_gen**2 * sigma_inv_full[None, :], axis=1)
        )
        rel_errs_gen = norm_diff_gen / norm_true_sigma
        mean_rel_err_gen = np.mean(rel_errs_gen)

        print("-" * 50)
        print(f"Sample {sample_idx} Error Metrics (Sigma^-1 norm, full 1080 timesteps):")
        print(f"  > Relative Sigma^-1 Error (q_mean vs q_true): {rel_err_mean_field:.6f} ({rel_err_mean_field * 100:.2f}%)")
        print(f"  > Mean Rel Sigma^-1 Error (q_gen  vs q_true): {mean_rel_err_gen:.6f} ({mean_rel_err_gen * 100:.2f}%)")
        print("-" * 50)


        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        timesteps = np.arange(1080)
        obs_timesteps = np.arange(0, 1080, 20)
        wells = ['Well 1', 'Well 2', 'Well 3', 'Well 4']

        for i in range(4):
            ax = axs[i // 2, i % 2]

            ax.plot(timesteps, q_gen_batch[:, i, :].T, color='gray', alpha=0.5, linewidth=1)
            ax.plot(timesteps, q_mean[i], color='blue', linestyle='--', linewidth=3.5)
            ax.plot(timesteps, q_true[i], color='red', linewidth=2.5, alpha=0.8)
            ax.scatter(obs_timesteps, q_obs[i], color='green', marker='o', s=30, zorder=10)

            ax.text(0.05, 0.9, wells[i], transform=ax.transAxes,
                    fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.set_title("")

            if i % 2 == 0:
                ax.set_ylabel('Oil Production Rate ($m^3/s$)', fontsize=16)

            if i // 2 == 1:
                ax.set_xlabel('Time Steps', fontsize=16)

            ax.tick_params(axis='both', which='major', labelsize=14)

            ax.grid(True, linestyle=':', alpha=0.5)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(formatter)

            ax.yaxis.get_offset_text().set_fontsize(14)


            if i == 0:
                custom_lines = [
                    Line2D([0], [0], color='gray', lw=1, alpha=1.0),
                    Line2D([0], [0], color='blue', ls='--', lw=3.5),
                    Line2D([0], [0], color='red', lw=2.5, alpha=0.8),
                    Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=6),
                ]
                legend_labels = ['Realizations', 'Mean', 'Ground Truth', 'Observations']

                ax.legend(custom_lines, legend_labels, loc='lower right', fontsize=12, frameon=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(OUTPUT_IMG_DIR, f"vis_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=150)
        print(f"  Saved plot to {save_path}")

        plt.close()

    print("\nAll plotting completed.")


if __name__ == "__main__":
    main()