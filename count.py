import os
import re
from collections import defaultdict

directory = r"C:\Users\nsrha\OneDrive\Desktop\ICNALE"

def count_files_by_score(directory):
    score_count = defaultdict(int)
    # Regular expression to match the scores (A1, A2, B1, B2, C1, C2)
    score_pattern = re.compile(r'_(A1|A2|B1|B2|C1|C2)_')

    # Iterate through files in the directory
    for file_name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file_name)):
            # Search for the score in the file name
            match = score_pattern.search(file_name)
            if match:
                score = match.group(1)
                # Increase the count for the score
                score_count[score] += 1

    # Print the count for each score
    for score, count in score_count.items():
        print(f"Score {score}: {count} files")

# Run the function
count_files_by_score(directory)
