import os
import re
import json
import argparse
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import random


def extract_sections(content: str) -> List[Dict[str, str]]:
    """
    Extracts sections from text content based on markers like \\title{, \\author{, \\section*{, etc.
    """
    section_patterns = [r'\\title\{', r'\\author\{', r'\\section\*\{', r'\\begin\{abstract\}']
    combined_pattern = '|'.join(section_patterns)
    section_starts = [(m.start(), m.group()) for m in re.finditer(f'({combined_pattern})', content)]
    
    if not section_starts:
        return [{"section_title": "content", "section_content": content.strip()}]
    
    sections = []
    
    for i in range(len(section_starts)):
        start_pos, marker = section_starts[i]
        
        if i < len(section_starts) - 1:
            end_pos = section_starts[i+1][0]
        else:
            end_pos = len(content)
        
        section_content = content[start_pos:end_pos].strip()
        
        if marker == '\\title{':
            title_match = re.search(r'\\title\{(.*?)\}', section_content, re.DOTALL)
            section_title = "title"
            if title_match:
                section_content = title_match.group(1).strip()
        elif marker == '\\author{':
            section_title = "author"
            author_match = re.search(r'\\author\{(.*?)\}', section_content, re.DOTALL)
            if author_match:
                section_content = author_match.group(1).strip()
        elif marker == '\\section*{':
            section_match = re.search(r'\\section\*\{(.*?)\}', section_content, re.DOTALL)
            if section_match:
                section_title = section_match.group(1).strip()
                content_start = re.search(r'\\section\*\{.*?\}', section_content, re.DOTALL).end()
                section_content = section_content[content_start:].strip()
            else:
                section_title = "unnamed_section"
        elif marker == '\\begin{abstract}':
            section_title = "abstract"
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', section_content, re.DOTALL)
            if abstract_match:
                section_content = abstract_match.group(1).strip()
        else:
            section_title = "other"
        
        sections.append({
            "section_title": section_title,
            "section_content": section_content
        })
    
    return sections


def extract_paper_title(content: str, filename: str) -> str:
    """
    Extract paper title from content or use filename if not found.
    """
    title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    if title_match:
        return title_match.group(1).strip()
    
    # If no title found, use the filename without extension
    return os.path.splitext(os.path.basename(filename))[0]


def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a single file and return structured data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paper_title = extract_paper_title(content, file_path)
    sections = extract_sections(content)
    
    return {
        "paper_title": paper_title,
        "sections": sections,
        "source_file": os.path.basename(file_path)
    }


def save_json(data: Dict[str, Any], output_path: str):
    """
    Save data as JSON to the specified path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_input(input_path: str, output_folder: str, prompts_data: Dict[str, Any] = None, max_tokens: int = 4000):
    """
    Process input (file or folder) and generate JSON output with section token counts and combinations.
    
    Args:
        input_path: Path to input file or folder
        output_folder: Path to output folder
        prompts_data: Optional dictionary containing contextual prompts data
        max_tokens: Maximum token limit for combined sections
    """
    input_path = os.path.abspath(input_path)
    output_folder = os.path.abspath(output_folder)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if os.path.isfile(input_path):
        # Process single file
        data = process_file(input_path)
        
        # If contextual prompts are provided, process sections with prompts
        if prompts_data:
            # Save the initial parsed data
            base_output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}.json")
            save_json(data, base_output_file)
            print(f"Processed {input_path} -> {base_output_file}")
            
            # Create temporary file with the parsed data
            temp_file = os.path.join(output_folder, "_temp_parsed.json")
            save_json(data, temp_file)
            
            # Process with prompts and save to a different file
            processed_data = process_paper_with_prompts(temp_file, prompts_data, max_tokens)
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_with_prompts.json")
            save_json(processed_data, output_file)
            print(f"Processed with prompts {input_path} -> {output_file}")
            
            # Remove temporary file
            try:
                os.remove(temp_file)
            except:
                pass
        else:
            # Just save the parsed data
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}.json")
            save_json(data, output_file)
            print(f"Processed {input_path} -> {output_file}")
            
    elif os.path.isdir(input_path):
        # Process all text files in the folder
        all_data = []
        all_processed_data = []
        
        for filename in os.listdir(input_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_path, filename)
                try:
                    data = process_file(file_path)
                    all_data.append(data)
                    print(f"Processed {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if all_data:
            # Save the parsed papers
            output_file = os.path.join(output_folder, f"{os.path.basename(input_path)}.json")
            save_json(all_data, output_file)
            print(f"Saved combined JSON to {output_file}")
            
            # Process with prompts if available
            if prompts_data:
                # Create temporary file for each paper and process it
                for i, paper_data in enumerate(all_data):
                    temp_file = os.path.join(output_folder, f"_temp_parsed_{i}.json")
                    save_json(paper_data, temp_file)
                    
                    try:
                        processed_data = process_paper_with_prompts(temp_file, prompts_data, max_tokens)
                        all_processed_data.extend(processed_data)
                    except Exception as e:
                        print(f"Error processing paper {i} with prompts: {e}")
                    
                    # Remove temporary file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                # Save the processed papers
                processed_output_file = os.path.join(output_folder, "parsed_papers_with_prompts.json")
                save_json(all_processed_data, processed_output_file)
                print(f"Saved combined processed JSON to {processed_output_file}")
        else:
            print("No valid text files found to process.")
    else:
        print(f"Input path {input_path} does not exist.")


def estimate_token_count(text: str) -> int:
    """
    Estimates the number of tokens in a given text string.
    This is a simple approximation based on common tokenization patterns.
    
    Args:
        text: The input text string
        
    Returns:
        Estimated token count
    """
    # Simple estimation methods:
    # 1. Whitespace-split words (underestimates)
    word_count = len(text.split())
    
    # 2. Character count divided by 4 (tokens are ~4 chars on average)
    char_count = len(text)
    char_based_estimate = char_count / 4
    
    # 3. Count punctuation and special characters as separate tokens
    punctuation = sum(1 for c in text if c in '.,!?;:()[]{}"\'\\/+-*=%$#@&^<>|~`')
    
    # Final estimate combines these approaches
    # Words + additional tokens for punctuation with a small adjustment factor
    estimate = word_count + punctuation
    
    # Apply a small correction factor based on comparison to char-based method
    if char_based_estimate > estimate * 1.3:
        estimate = (estimate + char_based_estimate) / 2
    
    return int(estimate)


def load_contextual_prompts(yaml_file: str) -> Dict[str, Any]:
    """
    Reads a YAML file containing contextual prompts, calculates the token count 
    for each prompt, and returns the data.

    Args:
        yaml_file: Path to the YAML file

    Returns:
        Dictionary containing prompts data with token counts
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'prompts' in data and isinstance(data['prompts'], list):
            for item in data['prompts']:
                if 'prompt' in item:
                    item['token_count'] = estimate_token_count(item['prompt'])
        
        return data
    except Exception as e:
        print(f"Error loading contextual prompts file {yaml_file}: {e}")
        return {}


def process_paper_with_prompts(paper_json_file: str, prompts_data: Dict[str, Any], max_tokens: int = 4000) -> List[Dict[str, Any]]:
    """
    Process paper sections, calculate token counts, and combine with appropriate contextual prompts.
    Each section is only used once in the output.
    
    Args:
        paper_json_file: Path to the paper JSON file
        prompts_data: Dictionary containing contextual prompts with token counts
        max_tokens: Maximum token limit (default: 4000)
        
    Returns:
        List of dictionaries containing section content with matched contextual prompts
    """
    with open(paper_json_file, 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    
    prompts_list = prompts_data.get('prompts', []) if prompts_data else []
    
    result_chunks = []
    
    if not prompts_list:
        sections = paper_data.get('sections', [])
        for i, section in enumerate(sections):
            section_title = section.get('section_title', 'Untitled Section')
            section_content = section.get('section_content', '')
            
            formatted_section = f"# {section_title}\n\n{section_content}"
            section_tokens = estimate_token_count(formatted_section)
            
            simplified_output = {
                'combined_content': formatted_section,
                'combined_tokens': section_tokens,
                'additional_sections': [],
                'used_prompt': None
            }
            
            result_chunks.append(simplified_output)
        return result_chunks
    
    used_section_indices = set()
    
    sections = paper_data.get('sections', [])
    i = 0
    while i < len(sections):
        if i in used_section_indices:
            i += 1
            continue
            
        section = sections[i]
        section_title = section.get('section_title', 'Untitled Section')
        section_content = section.get('section_content', '')
        
        formatted_section = f"# {section_title}\n\n{section_content}"
        section_tokens = estimate_token_count(formatted_section)
        
        compatible_prompts = []
        for prompt in prompts_list:
            if 'prompt' in prompt and 'token_count' in prompt:
                prompt_tokens = prompt['token_count']
                if section_tokens + prompt_tokens < max_tokens:
                    compatible_prompts.append(prompt)
        
        if compatible_prompts:
            current_section_data = {
                'section_title': section_title,
                'formatted_section': formatted_section,
                'section_tokens': section_tokens,
                'compatible_prompts': compatible_prompts
            }
            
            used_section_indices.add(i)
            
            next_i = i + 1
            combined_content = formatted_section
            combined_tokens = section_tokens
            additional_sections = []
            sections_used = [i]  # Keep track of sections used in this chunk
            
            while next_i < len(sections) and compatible_prompts:
                if next_i in used_section_indices:
                    next_i += 1
                    continue
                    
                next_section = sections[next_i]
                next_title = next_section.get('section_title', 'Untitled Section')
                next_content = next_section.get('section_content', '')
                
                formatted_next_section = f"\n\n# {next_title}\n\n{next_content}"
                next_tokens = estimate_token_count(formatted_next_section)
                
                still_compatible = []
                for prompt in compatible_prompts:
                    if combined_tokens + next_tokens + prompt['token_count'] < max_tokens:
                        still_compatible.append(prompt)
                
                if still_compatible:
                    combined_content += formatted_next_section
                    combined_tokens += next_tokens
                    additional_sections.append({
                        'section_title': next_title,
                        'section_tokens': next_tokens
                    })
                    compatible_prompts = still_compatible
                    
                    used_section_indices.add(next_i)
                    sections_used.append(next_i)
                    
                    next_i += 1
                else:
                    break
            
            selected_prompt = random.choice(compatible_prompts)
            
            final_content = f"{selected_prompt['prompt']}\n\n{combined_content}"
            final_tokens = combined_tokens + selected_prompt['token_count']
            
            simplified_output = {
                'combined_content': final_content,
                'combined_tokens': final_tokens,
                'additional_sections': additional_sections,
                'used_prompt': {
                    'id': selected_prompt.get('id', ''),
                    'prompt': selected_prompt['prompt'],
                    'token_count': selected_prompt['token_count']
                },
                'sections_used': sections_used  # For debugging, can be removed in production
            }
            
            result_chunks.append(simplified_output)
            i = next_i
        else:
            print(f"Section '{section_title}' with {section_tokens} tokens is too large for any prompt. Splitting...")
            
            used_section_indices.add(i)
            
            paragraphs = section_content.split('\n\n')
            if len(paragraphs) == 1:  # If no paragraphs, split by sentences
                paragraphs = re.split(r'(?<=[.!?])\s+', section_content)
            
            current_chunk = f"# {section_title} (Split)\n\n"
            current_tokens = estimate_token_count(current_chunk)
            sub_section_index = 0
            
            for paragraph in paragraphs:
                paragraph_tokens = estimate_token_count(paragraph)
                
                if current_tokens + paragraph_tokens < max_tokens / 2:  # Allow room for a prompt
                    current_chunk += paragraph + '\n\n'
                    current_tokens += paragraph_tokens
                else:
                    sub_compatible_prompts = []
                    for prompt in prompts_list:
                        if 'prompt' in prompt and 'token_count' in prompt:
                            prompt_tokens = prompt['token_count']
                            if current_tokens + prompt_tokens < max_tokens:
                                sub_compatible_prompts.append(prompt)
                    
                    if sub_compatible_prompts:
                        selected_prompt = random.choice(sub_compatible_prompts)
                        
                        final_content = f"{selected_prompt['prompt']}\n\n{current_chunk.strip()}"
                        final_tokens = current_tokens + selected_prompt['token_count']
                        
                        simplified_output = {
                            'combined_content': final_content,
                            'combined_tokens': final_tokens,
                            'additional_sections': [],
                            'used_prompt': {
                                'id': selected_prompt.get('id', ''),
                                'prompt': selected_prompt['prompt'],
                                'token_count': selected_prompt['token_count']
                            },
                            'section_split': True  # Indicate this is a split section
                        }
                        
                        result_chunks.append(simplified_output)
                    
                    sub_section_index += 1
                    current_chunk = f"# {section_title} (Part {sub_section_index + 1})\n\n{paragraph}\n\n"
                    current_tokens = estimate_token_count(current_chunk)
            
            if current_chunk.strip(): # Check if current_chunk has content beyond the title part
                sub_compatible_prompts = []
                for prompt in prompts_list:
                    if 'prompt' in prompt and 'token_count' in prompt:
                        prompt_tokens = prompt['token_count']
                        if current_tokens + prompt_tokens < max_tokens:
                            sub_compatible_prompts.append(prompt)
                
                if sub_compatible_prompts:
                    selected_prompt = random.choice(sub_compatible_prompts)
                    
                    final_content = f"{selected_prompt['prompt']}\n\n{current_chunk.strip()}"
                    final_tokens = current_tokens + selected_prompt['token_count']
                    
                    simplified_output = {
                        'combined_content': final_content,
                        'combined_tokens': final_tokens,
                        'additional_sections': [],
                        'used_prompt': {
                            'id': selected_prompt.get('id', ''),
                            'prompt': selected_prompt['prompt'],
                            'token_count': selected_prompt['token_count']
                        },
                        'section_split': True  # Indicate this is a split section
                    }
                    
                    result_chunks.append(simplified_output)
            
            i += 1
    
    # Remove debugging fields if needed
    for chunk in result_chunks:
        if 'sections_used' in chunk:
            del chunk['sections_used']
        if 'section_split' in chunk:
            del chunk['section_split']
    
    return result_chunks


def main():
    parser = argparse.ArgumentParser(description='Parse text files and convert to JSON format.')
    parser.add_argument('--input', help='Input file or folder')
    parser.add_argument('--output', help='Output folder')
    parser.add_argument('--context', help='Contextual prompy yaml file')
    parser.add_argument('--max-tokens', type=int, default=4000, help='Maximum token limit (default: 4000)')
    args = parser.parse_args()
    
    prompts_data = None
    if args.context:
        prompts_data = load_contextual_prompts(args.context)
    
    if args.input and args.output:
        process_input(args.input, args.output, prompts_data, args.max_tokens)


if __name__ == "__main__":
    main()
