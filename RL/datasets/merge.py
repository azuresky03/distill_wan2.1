import os
import argparse

def merge_txt_files(input_dir, output_dir, output_filename="merged.txt"):
    """
    Merge all .txt files in input_dir into a single file in output_dir.
    
    Args:
        input_dir (str): Path to directory containing .txt files to merge
        output_dir (str): Path to directory where the merged file will be saved
        output_filename (str, optional): Name of the output file. Defaults to "merged.txt".
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .txt files in input directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    # Merge contents
    merged_content = ""
    for filename in txt_files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            merged_content += content
            # Add a newline if the file doesn't end with one
            if not content.endswith('\n'):
                merged_content += '\n'
    
    # Write merged content to output file
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(merged_content)
    
    print(f"Successfully merged {len(txt_files)} files into {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Merge multiple text files into one.')
    parser.add_argument('--input_dir', default='/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets/valid_data_t2v', help='Directory containing text files to merge')
    parser.add_argument('--output_dir', default='/cv/zhangpengpeng/cv/video_generation/Wan2.1/RL/datasets', help='Directory to save the merged file')
    parser.add_argument('--output_filename', default='merged.txt', help='Name of the output merged file')
    
    args = parser.parse_args()
    
    merge_txt_files(args.input_dir, args.output_dir, args.output_filename)

if __name__ == "__main__":
    main()