import subprocess
import sys
import time
import os


CONFIG = {
    "flow_test":       ".../raw_data/subqwt_last200_noisy.txt",
    "ckpt_path":       ".../saved_model/flow_matching_best.pt",
    "save_dir":        ".../perms_file_name",
    "output_img_dir":  ".../vis_file_name",
    "output_data_dir": ".../production_file_name",
    "n_test":     3,     # number of test cases
    "n_times":    2000,  # number of samples generated in total
    "chunk_size": 50,    # number of samples generated each time
    "eta_noise":  0.03,  # noise level
    "net_module": "module",
    "net_class":  "UNetCond3D",
    "use_guidance":   1,    # 1 for guidance, 0 for non-guidance
    "guidance_scale": 1.0,
    "clamp_ratio":    0.05, # parameter controlling the guidance strength
    "start_sample":   0

}

SCRIPTS = [
    # Step 1 Sampling
    ".../flow_matching.py",
    
    # Step 2 Simulator Calculation
    ".../qwt_computation.py",
    
    # Step 3 Plotting
    ".../plot_saved_qwt.py"
]


def run_script(script_path, step_num, total_steps):
    if not os.path.exists(script_path):
        print(f"\n[Error] File not found: {script_path}")
        return False

    script_name = os.path.basename(script_path)
    print("=" * 60)
    print(f"[{step_num}/{total_steps}] Running: {script_name}")
    print(f"Path: {script_path}")
    print("=" * 60)

    start_time = time.time()
    cmd = [sys.executable, script_path]
    if "flow_matching" in script_name:
        print(f"-> Passing dynamic arguments to {script_name}...")
        cmd.extend([
            "--flow_test",      CONFIG["flow_test"],
            "--ckpt_path",      CONFIG["ckpt_path"],
            "--save_dir",       CONFIG["save_dir"],
            "--n_test",         str(CONFIG["n_test"]),
            "--n_times",        str(CONFIG["n_times"]),
            "--chunk_size",     str(CONFIG["chunk_size"]),
            "--eta_noise",      str(CONFIG["eta_noise"]),
            "--net_module",     CONFIG["net_module"],
            "--net_class",      CONFIG["net_class"],
            "--use_guidance",   str(CONFIG["use_guidance"]),
            "--guidance_scale", str(CONFIG["guidance_scale"]),
            "--clamp_ratio",    str(CONFIG["clamp_ratio"])

        ])
    elif "qwt_computation" in script_name:
        print(f"-> Passing arguments to QWT Computation script...")
        cmd.extend([
            "--saved_txt_dir",   CONFIG["save_dir"],
            "--obs_file",        CONFIG["flow_test"],
            "--output_img_dir",  CONFIG["output_img_dir"],
            "--output_data_dir", CONFIG["output_data_dir"],
            "--start_sample",    str(CONFIG["start_sample"])
        ])
    elif "plot_saved_qwt" in script_name:
        print(f"-> Passing arguments to Step 3 (Final Plotting)...")
        cmd.extend([
            "--pred_data_dir",    CONFIG["output_data_dir"],            
            "--perm_samples_dir", CONFIG["save_dir"],
            "--obs_file",         CONFIG["flow_test"],
            "--output_img_dir",   CONFIG["output_img_dir"]
        ])


    try:
        print(f"Executing command: {' '.join(cmd)}") 
        subprocess.run(cmd, check=True)      
        elapsed = time.time() - start_time
        print(f"\n Finished: {script_name} (Time: {elapsed:.2f}s)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n Error: '{script_name}' failed with exit code {e.returncode}.")
        return False
    except KeyboardInterrupt:
        print("\n Pipeline interrupted by user.")
        return False

def main():
    print(f"Starting Pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Interpreter: {sys.executable}")
    
    total_scripts = len(SCRIPTS)
    
    for i, script_path in enumerate(SCRIPTS):
        success = run_script(script_path, i + 1, total_scripts)
        
        if not success:
            print("\n Pipeline stopped due to error.")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()