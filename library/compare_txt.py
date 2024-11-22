def compare_files(file1_path, file2_path):

    with open(file1_path, 'r') as file1:
        file1_lines = file1.readlines()

    with open(file2_path, 'r') as file2:
        file2_lines = file2.readlines()

    max_lines = max(len(file1_lines), len(file2_lines))

    differences = []

    for i in range(max_lines):
        line1 = file1_lines[i].strip() if i < len(file1_lines) else ''
        line2 = file2_lines[i].strip() if i < len(file2_lines) else ''

        if line1 != line2:
            differences.append(f"Line {i + 1}:\nFile 1: {line1}\nFile 2: {line2}\n")

    if differences:
        print("Differences found:")
        for diff in differences:
            print(diff)
    else:
        print("No differences found.")

file1_path = '../model/channel_info_base.txt'
file2_path = '../model/channel_info_depth.txt'
compare_files(file1_path, file2_path)
