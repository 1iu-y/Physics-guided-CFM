import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import argparse
import sys
from tqdm import tqdm
src_path = ".../src"
if src_path not in sys.path:
    sys.path.append(src_path)
from src.fvm_twophase_batch import twophase_impes_torch


PERM_GT_FILE = '.../raw_data/perms3d_trans.txt'
FLOW_GT_FILE = '.../raw_data/q_total.txt'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SIM_BATCH_SIZE = 32

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1



def run_simulator_in_batches(perms_flat_tensor, batch_size=32):
    N = perms_flat_tensor.shape[0]
    results_list = []

    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc=f"  Simulating ({N} samples)", leave=False):
            batch_perms = perms_flat_tensor[i: i + batch_size]
            B_curr = batch_perms.shape[0]

            perms_3d = batch_perms.view(B_curr, 1, 5, 20, 20)
            perms_phys = (2.0 ** perms_3d.clamp(min=-3.5, max=10.5)) * 1e-16
            perms_phys = perms_phys.to(DEVICE).to(torch.float64)

            q_out = twophase_impes_torch(perms_phys, return_all=False, device=DEVICE)

            if isinstance(q_out, tuple):
                q_out = q_out[0]

            q_reshaped = q_out.view(B_curr, 4, 1080).cpu()
            results_list.append(q_reshaped)

    final_q = torch.cat(results_list, dim=0).numpy()
    return final_q


def extract_sample_id(filename):
    match = re.search(r'sample_(\d+).txt', filename)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser(description="QWT Computation & Visualization")

    parser.add_argument("--saved_txt_dir", type=str, default='.../perms_file_name',
                        help="Input directory containing generated permeability samples")
    
    parser.add_argument("--obs_file", type=str, default='.../raw_data/subqwt_last200_noisy_6.5percent.txt',
                        help="Path to observation data file")
    
    parser.add_argument("--output_img_dir", type=str, default='.../vis_file_name',
                        help="Directory to save visualization images")
    
    parser.add_argument("--output_data_dir", type=str, default='.../production_file_name',
                        help="Directory to save computed flow rate data")
    
    parser.add_argument("--start_sample", type=int, default=0,
                        help="Starting sample index to process")

    args = parser.parse_args()

    SAVED_TXT_DIR = args.saved_txt_dir
    OBS_FILE = args.obs_file
    OUTPUT_IMG_DIR = args.output_img_dir
    OUTPUT_DATA_DIR= args.output_data_dir
    START_SAMPLE = args.start_sample


    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.makedirs(OUTPUT_DATA_DIR)

    print("Loading Ground Truth Permeability...")
    if not os.path.exists(PERM_GT_FILE):
        print(f"Error: {PERM_GT_FILE} not found.")
        return
    all_perms_np = np.loadtxt(PERM_GT_FILE)
    N_total = all_perms_np.shape[0]

    n_train = int(N_total * TRAIN_RATIO)
    n_val = int(N_total * VAL_RATIO)
    test_start_index = n_train + n_val
    print(f"Test set starts at index: {test_start_index}")

    print(f"Loading Ground Truth Flow Data from {FLOW_GT_FILE}...")
    if not os.path.exists(FLOW_GT_FILE):
        print(f"Error: {FLOW_GT_FILE} not found.")
        return
													   
    all_flow_gt_np = np.loadtxt(FLOW_GT_FILE)

    print(f"Loading Observation Data from {OBS_FILE}...")
    if not os.path.exists(OBS_FILE):
        print(f"Error: {OBS_FILE} not found.")
        return
    all_obs_np = np.loadtxt(OBS_FILE)

    txt_files = glob.glob(os.path.join(SAVED_TXT_DIR, "sample_*.txt"))
    txt_files.sort(key=extract_sample_id)

    if not txt_files:
        print(f"No files found in directory: {SAVED_TXT_DIR}")
        return

    print(f"Found {len(txt_files)} samples. Starting processing...")

    pbar = tqdm(txt_files, desc="Total Progress")

    
    
    for txt_file in pbar:
        sample_idx = extract_sample_id(txt_file)

        if sample_idx < START_SAMPLE:
            continue

        pbar.set_description(f"Processing Sample {sample_idx}")

        try:
            gen_perms_np = np.loadtxt(txt_file)
        except Exception as e:
            tqdm.write(f"Error loading {txt_file}: {e}")
            continue

        if len(gen_perms_np.shape) == 1:
            gen_perms_np = gen_perms_np[np.newaxis, :]

        n_times = gen_perms_np.shape[0]

        global_idx = test_start_index + sample_idx

        if global_idx >= N_total:
            tqdm.write("GT index out of bounds, skipping.")
            continue

															   
        if global_idx >= all_flow_gt_np.shape[0]:
            tqdm.write(f"Flow GT index {global_idx} out of bounds, skipping.")
            continue

        q_true_1080 = all_flow_gt_np[global_idx].reshape(4, 1080)
        if sample_idx >= all_obs_np.shape[0]:
            tqdm.write("Observation index out of bounds, skipping.")
            continue
        obs_row = all_obs_np[sample_idx]
        obs_data = obs_row.reshape(4, 54)

        q_gen_batch = run_simulator_in_batches(
            torch.from_numpy(gen_perms_np),
            batch_size=SIM_BATCH_SIZE
        )

        q_save_path = os.path.join(OUTPUT_DATA_DIR, f"q_pred_sample_{sample_idx}.txt")
        q_gen_flat = q_gen_batch.reshape(n_times, -1)
        np.savetxt(q_save_path, q_gen_flat, fmt='%.6e', delimiter=' ')

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Flow Rate Analysis: Sample {sample_idx} (n={n_times})', fontsize=16)

							
        timesteps = np.arange(1080)

									
        obs_timesteps = np.arange(0, 1080, 20)

        wells = ['Well 1', 'Well 2', 'Well 3', 'Well 4']

        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.plot(timesteps, q_gen_batch[:, i, :].T,
                    color='gray', alpha=0.9, linewidth=1)
            q_mean = np.mean(q_gen_batch[:, i, :], axis=0)
            ax.plot(timesteps, q_mean, color='blue', linestyle='--', linewidth=3.5)
            ax.plot(timesteps, q_true_1080[i], color='red', linewidth=2.5, alpha=0.8)
            ax.scatter(obs_timesteps, obs_data[i], color='green', marker='o', s=30, zorder=10)


            ax.set_title(wells[i])
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Flow Rate')
            ax.grid(True, linestyle=':', alpha=0.5)

            if i == 0:
                from matplotlib.lines import Line2D
                custom = [
                    Line2D([0], [0], color='gray', lw=1, alpha=0.05),
                    Line2D([0], [0], color='blue', ls='--', lw=2),
                    Line2D([0], [0], color='red', lw=2.5),
                    Line2D([0], [0], color='green', marker='o', linestyle='None', markersize=6)
													  
                ]
                ax.legend(custom, ['Realizations', 'Mean', 'Ground Truth', 'Observation'])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_img_path = os.path.join(OUTPUT_IMG_DIR, f"vis_sample_{sample_idx}.png")
        plt.savefig(save_img_path, dpi=150)
												 

        plt.close()

    print("\nAll processing completed.")


if __name__ == "__main__":
    main()