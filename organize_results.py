# organize_results.py
import os
import shutil

# --- Configuration ---
# The script assumes it's being run from the 'vllm_benchmarking' directory.
source_directory = 'results'
destination_directory = 'no_duplicates'


def organize_benchmark_files():
    """
    Finds unique benchmark results, creates a new directory,
    and copies a single file for each unique run into it.
    """
    # --- Step 1: Create the destination directory ---
    try:
        os.makedirs(destination_directory, exist_ok=True)
        print(f"Directory '{destination_directory}' is ready.")
    except OSError as e:
        print(f"Error: Could not create directory {destination_directory}: {e}")
        return

    # --- Step 2: Get all files from the source directory ---
    try:
        all_files = os.listdir(source_directory)
    except FileNotFoundError:
        print(f"Error: Source directory '{source_directory}' not found.")
        print("Please make sure you are in the 'vllm_benchmarking' directory before running.")
        return

    # --- Step 3: Identify the unique base filenames ---
    unique_basenames = set()
    for filename in all_files:
        if filename.endswith('.json'):
            # Remove both possible suffixes to get the unique base name
            base_name = filename.replace('_input_fixed.json', '').replace('_output_fixed.json', '')
            unique_basenames.add(base_name)

    print(f"\nFound {len(all_files)} total files, representing {len(unique_basenames)} unique benchmarks.")

    # --- Step 4: Copy one version of each unique file ---
    copied_count = 0
    print("Starting copy process...")
    # Sort the list for a clean, ordered output
    for base_name in sorted(list(unique_basenames)):
        # We'll consistently copy the '_input_fixed.json' file for each run.
        source_file_to_copy = f"{base_name}_input_fixed.json"
        source_path = os.path.join(source_directory, source_file_to_copy)

        # Give the new file a cleaner name in the destination
        clean_destination_name = f"{base_name}.json"
        destination_path = os.path.join(destination_directory, clean_destination_name)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, destination_path)
                copied_count += 1
            except Exception as e:
                print(f"  - Could not copy {source_path}: {e}")
        else:
            # This handles cases where a file might be missing
            print(f"  - Warning: Source file not found: {source_path}")

    print(f"\nProcess complete. Copied {copied_count} files to '{destination_directory}'.")


if __name__ == "__main__":
    organize_benchmark_files()
