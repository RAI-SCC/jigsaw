import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd

def extract_data(path: str, matrix_size: list[int]):
    lines = None
    data = {}
    k = 0

    with open(path, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if "[SUMMARY]" in lines[i]:
            data[matrix_size[k]] = eval(lines[i + 1])
            k += 1

    return data

def extract_files_data(files: list[str], world_size: list[int], matrix_size: list[int]):
    data = {}
    i = 0

    for f in files:
        data[world_size[i]] = extract_data(f, matrix_size)

        i += 1

    return data

def data_to_matrix_index(old_data: dict):
    new_data = {}

    for world_size, matrix_dict in old_data.items():
        for matrix_size, alg_dict in matrix_dict.items():
            if matrix_size not in new_data:
                new_data[matrix_size] = {}
            if world_size not in new_data[matrix_size]:
                new_data[matrix_size][world_size] = {}
            for algorithm, value in alg_dict.items():
                new_data[matrix_size][world_size][algorithm] = value

    return new_data

def plot_scalability_results(data, output_dir="scaling_results"):    
# Restructure: data[matrix_size][world_size][alg_key] 
    # -> parsed[matrix_size][trans_type][base_alg][world_size] = time
    parsed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for matrix_size, world_dict in data.items():
        for world_size, alg_dict in world_dict.items():
            for alg_key, time_val in alg_dict.items():
                # Split key like "SUMMA_X^T@W" into base algorithm and transposition type
                if "_" in alg_key:
                    base_alg, trans_type = alg_key.split("_", 1)
                else:
                    base_alg, trans_type = alg_key, "X@W"
                    
                parsed_data[matrix_size][trans_type][base_alg][world_size] = time_val
                
    # Generate a separate plot for each matrix size and transposition combination
    for matrix_size, trans_dict in parsed_data.items():
        for trans_type, alg_data in trans_dict.items():
            plt.figure(figsize=(8, 5))
            
            for base_alg, world_time_dict in alg_data.items():
                sorted_worlds = sorted(world_time_dict.keys())
                x_vals = sorted_worlds
                y_vals = [world_time_dict[w] for w in sorted_worlds]
                
                plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=base_alg)
            
            plt.title(f"Matrix Size: {matrix_size} | Operation: {trans_type}")
            plt.xlabel("World Size (Core Count)")
            plt.ylabel("Execution Time (seconds)")
            plt.legend(title="Algorithms")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            
            # Create a safe filename by replacing special characters (@, ^)
            safe_trans = trans_type.replace("@", "_").replace("^", "")
            filename = os.path.join(output_dir, f"scalability_{matrix_size}_{safe_trans}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

def convert_results_to_tables(data):
    parsed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for matrix_size, world_dict in data.items():
        for world_size, alg_dict in world_dict.items():
            for alg_key, time_val in alg_dict.items():
                if "_" in alg_key:
                    base_alg, trans_type = alg_key.split("_", 1)
                else:
                    base_alg, trans_type = alg_key, "X@W"
                    
                parsed_data[matrix_size][trans_type][base_alg][world_size] = time_val
                
    tables = {}
    for matrix_size, trans_dict in parsed_data.items():
        for trans_type, alg_data in trans_dict.items():
            df = pd.DataFrame(alg_data).T.sort_index()
            df.index.name = "Algorithm"
            df.columns.name = "World Size (Cores)"
            tables[(matrix_size, trans_type)] = df
            
    return tables

if __name__ == "__main__":
    file_path = r"scaling_results/"
    files = [file_path + "ALL-1-refactored.out", file_path + "ALL-4-refactored.out", file_path + "ALL-16-refactored.out"]
    ws = [1, 4, 16]
    mats = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    data_world = extract_files_data(files, ws, mats)
    data_mat = data_to_matrix_index(data_world)

    # plot_scalability_results(data_mat)

    table = convert_results_to_tables(data_mat)

    print(table.keys())
    print(table[(32768, "X@W")])
    print(table[(32768, "X^T@W")])
    print(table[(32768, "X@W^T")])