import os

def process_files(input_dir, output_dir, start_lines):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
            # Process each line in the input file
            for line in input_file:
                if any(line.strip().startswith(start_line) for start_line in start_lines):
                    output_file.write(line)
# Example usage
input_directory = 'fixes_in'
output_directory = 'fixes_out'
start_lines = ['heuristic', 'running', 'starter', 'saved', 'cost']

process_files(input_directory, output_directory, start_lines)
