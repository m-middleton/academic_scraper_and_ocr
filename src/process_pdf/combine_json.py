#!/usr/bin/env python3
import json
import argparse
import os
import glob
import collections
import statistics
from typing import List, Dict, Any, Union


def load_json_file(filepath: str) -> Union[List[Any], Dict[str, Any]]:
    """Load and parse a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_json_files(directory: str) -> List[str]:
    """Find all JSON files in the specified directory."""
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory")
    
    json_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"Warning: No JSON files found in '{directory}'")
    else:
        print(f"Found {len(json_files)} JSON files in '{directory}'")
    
    return json_files


def generate_metrics(data: List[Any], metrics_file: str) -> None:
    """
    Generate metrics about the combined JSON data and write them to a text file.
    """
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Total number of items: {len(data)}\n")
        
        total_size_bytes = len(json.dumps(data))
        size_kb = total_size_bytes / 1024
        size_mb = size_kb / 1024
        f.write(f"Total size: {total_size_bytes} bytes ({size_kb:.2f} KB, {size_mb:.2f} MB)\n\n")
        
        item_types = collections.Counter()
        field_counts = collections.Counter()
        all_keys = set()
        
        combined_content_count = 0
        
        combined_tokens_count = 0
        combined_tokens_values = []
        
        used_prompt_counts = collections.Counter()
        
        for item in data:
            item_type = type(item).__name__
            item_types[item_type] += 1
            
            if isinstance(item, dict):
                field_counts[len(item)] += 1
                all_keys.update(item.keys())
                
                if 'combined_content' in item:
                    combined_content_count += 1
                
                if 'combined_tokens' in item:
                    combined_tokens_count += 1
                    if isinstance(item['combined_tokens'], (int, float)):
                        combined_tokens_values.append(item['combined_tokens'])
                
                if 'used_prompt' in item:
                    prompt_value = str(item['used_prompt'])
                    used_prompt_counts[prompt_value] += 1
        
        f.write("Item Types:\n")
        for item_type, count in item_types.items():
            percentage = (count / len(data)) * 100
            f.write(f"- {item_type}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        if 'dict' in item_types:
            f.write("Field Count Distribution (for dict items):\n")
            total_dicts = item_types['dict']
            for field_count, count in sorted(field_counts.items()):
                percentage = (count / total_dicts) * 100
                f.write(f"- {field_count} fields: {count} items ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write(f"Total unique fields across all dictionary items: {len(all_keys)}\n")
            if len(all_keys) <= 20:
                f.write(f"All fields: {', '.join(sorted(all_keys))}\n")
            else:
                f.write(f"First 20 fields (alphabetical): {', '.join(sorted(list(all_keys)[:20]))}\n")
            f.write("\n")
        
        f.write(f"Items with combined_content: {combined_content_count} ({(combined_content_count / len(data)) * 100:.1f}%)\n\n")
        
        f.write(f"Items with combined_tokens: {combined_tokens_count} ({(combined_tokens_count / len(data)) * 100:.1f}%)\n")
        
        if combined_tokens_values:
            min_tokens = min(combined_tokens_values)
            max_tokens = max(combined_tokens_values)
            avg_tokens = statistics.mean(combined_tokens_values)
            
            f.write(f"combined_tokens statistics:\n")
            f.write(f"- Min tokens: {min_tokens}\n")
            f.write(f"- Max tokens: {max_tokens}\n")
            f.write(f"- Average tokens: {avg_tokens:.1f}\n")
            
            token_ranges = [(0, 100), (101, 500), (501, 1000), (1001, 5000), (5001, float('inf'))]
            f.write(f"Token count distribution:\n")
            for start, end in token_ranges:
                end_display = "inf" if end == float('inf') else end
                count = sum(1 for t in combined_tokens_values if start <= t <= end)
                percentage = (count / len(combined_tokens_values)) * 100
                f.write(f"- {start}-{end_display} tokens: {count} ({percentage:.1f}%)\n")
            f.write("\n")
        
        if used_prompt_counts:
            f.write(f"used_prompt distribution (all {len(used_prompt_counts)} values):\n")
            for prompt, count in used_prompt_counts.most_common():
                percentage = (count / len(data)) * 100
                # Truncate long prompts for display but still show full count
                display_prompt = prompt if len(prompt) < 50 else prompt[:47] + "..."
                f.write(f"- '{display_prompt}': {count} ({percentage:.1f}%)\n")
            f.write("\n")
                
    print(f"Metrics written to {metrics_file}")


def combine_json_files(input_files: List[str], output_file: str) -> List[Any]:
    """
    Combine multiple JSON files into one file.
    Handles both JSON objects (dicts) and arrays (lists).
    Returns the combined data.
    """
    combined_data = []
    processed_count = 0
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue
            
        try:
            data = load_json_file(file_path)
            
            if isinstance(data, list):
                combined_data.extend(data)
                processed_count += 1
            elif isinstance(data, dict):
                combined_data.append(data)
                processed_count += 1
            else:
                print(f"Warning: Unexpected data type in {file_path}, skipping.")
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse {file_path} as JSON, skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Successfully combined {processed_count} JSON files into {output_file}")
    
    return combined_data


def process_directory(directory: str, output_file: str, metrics_file: str = None) -> None:
    """Process all JSON files in a directory and combine them."""
    try:
        json_files = find_json_files(directory)
        if not json_files:
            return
        
        combined_data = combine_json_files(json_files, output_file)
        
        if metrics_file:
            generate_metrics(combined_data, metrics_file)
            
    except Exception as e:
        print(f"Error processing directory '{directory}': {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Combine JSON files from a directory into one file.')
    parser.add_argument('--input-folder', help='Directory containing JSON files to combine')
    parser.add_argument('-o', '--output', default='combined_output.json', 
                        help='Output JSON file (default: combined_output.json)')
    parser.add_argument('-m', '--metrics', default=None,
                        help='Generate metrics about the combined JSON and save to this file')
    
    args = parser.parse_args()
    process_directory(args.input_folder, args.output, args.metrics)


if __name__ == "__main__":
    main()
