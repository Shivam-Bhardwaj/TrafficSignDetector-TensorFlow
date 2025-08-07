#!/usr/bin/env python3
"""
Emoji detection script to prevent emojis in code and documentation.
This script scans all relevant files and fails if emojis are found.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

# Common emoji Unicode ranges
EMOJI_PATTERNS = [
    r'[\U0001F600-\U0001F64F]',  # emoticons
    r'[\U0001F300-\U0001F5FF]',  # symbols & pictographs
    r'[\U0001F680-\U0001F6FF]',  # transport & map
    r'[\U0001F1E0-\U0001F1FF]',  # flags (iOS)
    r'[\U00002600-\U000026FF]',  # miscellaneous symbols
    r'[\U00002700-\U000027BF]',  # dingbats
    r'[\U0001F900-\U0001F9FF]',  # supplemental symbols and pictographs
    r'[\U0001FA70-\U0001FAFF]',  # symbols and pictographs extended-A
]

EMOJI_REGEX = re.compile('|'.join(EMOJI_PATTERNS))

def find_emojis_in_file(file_path: Path) -> List[Tuple[int, str]]:
    """Find emojis in a file and return line numbers and content."""
    emojis_found = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if EMOJI_REGEX.search(line):
                    emojis_found.append((line_num, line.strip()))
    except UnicodeDecodeError:
        # Skip binary files
        pass
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    
    return emojis_found

def check_project_for_emojis(project_root: Path) -> bool:
    """Check entire project for emojis. Returns True if emojis found."""
    files_to_check = []
    
    # File patterns to check
    patterns = [
        "*.py",
        "*.md", 
        "*.yml",
        "*.yaml",
        "*.txt",
        "*.ini",
        "*.cfg",
        "*.json"
    ]
    
    # Directories to skip
    skip_dirs = {
        '.git', '__pycache__', '.tox', 'venv', '.venv', 
        'node_modules', 'build', 'dist', '.eggs'
    }
    
    for pattern in patterns:
        for file_path in project_root.rglob(pattern):
            # Skip files in excluded directories
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            files_to_check.append(file_path)
    
    emojis_found = False
    total_files_checked = 0
    
    for file_path in files_to_check:
        total_files_checked += 1
        emoji_lines = find_emojis_in_file(file_path)
        
        if emoji_lines:
            emojis_found = True
            print(f"EMOJI DETECTED in {file_path}:")
            for line_num, line_content in emoji_lines:
                # Handle encoding issues by replacing problematic characters
                safe_content = line_content.encode('ascii', errors='replace').decode('ascii')
                print(f"  Line {line_num}: {safe_content}")
            print()
    
    print(f"Checked {total_files_checked} files for emojis.")
    
    if emojis_found:
        print("ERROR: Emojis found in codebase!")
        print("Please remove all emojis before committing.")
        return True
    else:
        print("SUCCESS: No emojis found in codebase.")
        return False

def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    
    print("Scanning project for emojis...")
    print(f"Project root: {project_root}")
    print()
    
    emojis_found = check_project_for_emojis(project_root)
    
    if emojis_found:
        sys.exit(1)  # Fail CI/CD pipeline
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()