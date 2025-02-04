import os

# Define the folder path and output file
folder_path = r"C:/Python Projects/gaming-bot"
output_file = "all_raw_code.txt"

# Create a list of folders to exclude
excluded_folders = [
    os.path.join(folder_path, "scripts"),
    os.path.join(folder_path, "models")
]

# Open the output file for writing
with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(folder_path):
        # Skip if the current root starts with any of the excluded folders
        if any(root.startswith(exclude_path) for exclude_path in excluded_folders):
            continue

        for file in files:
            if file.endswith(".py"):
                # Get the full path to the .py file
                file_path = os.path.join(root, file)
                
                # Write a separator and the file path
                outfile.write(f"\n{'-'*80}\n")
                outfile.write(f"File Path: {file_path}\n")
                outfile.write(f"{'-'*80}\n\n")
                
                # Read the .py file's content and write it to the output file
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")

print(f"All Python files (excluding scripts and model folders) have been combined into {output_file}.")
