#!/usr/bin/env python
import os
import argparse
import re
from typing import Callable, List, Union, Dict
from pathlib import Path


def read_file(file_path: str) -> str:
    """
    Read the content of a text file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        The content of the file as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def write_file(output_path: str, content: str) -> None:
    """
    Write content to a text file.
    
    Args:
        output_path: Path where the file should be written
        content: String content to write to the file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)


def parse_text(text: str, parsing_functions: List[Callable[[str], str]]) -> str:
    """
    Apply a list of parsing functions to the text.
    
    Args:
        text: The input text to parse
        parsing_functions: A list of functions to apply to the text
    
    Returns:
        The parsed text after applying all parsing functions
    """
    result = text
    for func in parsing_functions:
        result = func(result)
    return result


def process_file(input_path: str, output_path: str, parsing_functions: List[Callable[[str], str]]) -> None:
    """
    Process a single file by reading it, parsing its content, and writing the result.
    
    Args:
        input_path: Path to the input file
        output_path: Path to the output file
        parsing_functions: List of functions to apply to the text
    """
    text = read_file(input_path)
    parsed_text = parse_text(text, parsing_functions)
    write_file(output_path, parsed_text)


def process_files(input_path: Union[str, Path], output_folder: Union[str, Path], 
                 parsing_functions: List[Callable[[str], str]]) -> None:
    """
    Process one or multiple files based on the input path.
    
    Args:
        input_path: Path to a file or directory containing files to process
        output_folder: Path to the directory where processed files will be saved
        parsing_functions: List of functions to apply to the text
    """
    input_path = Path(input_path)
    output_folder = Path(output_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    
    if input_path.is_file():
        output_file = output_folder / input_path.name
        process_file(str(input_path), str(output_file), parsing_functions)
        print(f"Processed: {input_path} -> {output_file}")
    
    elif input_path.is_dir():
        for file_path in input_path.glob('*.txt'):
            output_file = output_folder / file_path.name
            process_file(str(file_path), str(output_file), parsing_functions)
            print(f"Processed: {file_path} -> {output_file}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


def remove_footnotes(text: str) -> str:
    """
    Remove LaTeX footnotes from text.
    
    Args:
        text: Input text containing LaTeX footnotes
    
    Returns:
        Text with footnotes removed
    """
    # Match LaTeX footnote commands and their content
    pattern = r'\\footnotetext\{[^}]*\}'
    return re.sub(pattern, '', text)


def remove_author_sections(text: str) -> str:
    """
    Remove author sections from LaTeX documents.
    
    Args:
        text: Input text containing LaTeX author sections
    
    Returns:
        Text with author sections removed
    """
    # Pattern to match \author{...} and all its contents
    pattern = r'\\author\{(?:.*?\n)*?.*?\}'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_acknowledgment_sections(text: str) -> str:
    """
    Remove acknowledgment sections from text.
    
    Args:
        text: Input text containing acknowledgment sections
    
    Returns:
        Text with acknowledgment sections removed
    """
    # Case-insensitive pattern for acknowledgment sections
    pattern = r'\\section\*\{[Aa][Cc][Kk][Nn][Oo][Ww][Ll][Ee][Dd][Gg](?:[Ee]?[Mm][Ee][Nn][Tt]|[Ee][Mm][Ee][Nn][Tt])s?\}(?:.*?\n)(?:(?!\\section\*\{).*?\n)*'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_reference_sections(text: str) -> str:
    """
    Remove reference sections from text and all content after it.
    
    Args:
        text: Input text containing reference sections
    
    Returns:
        Text with reference sections and all following content removed
    """
    # Find the references section and remove it and everything after it
    pattern = r'\\section\*\{[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss]\}.*'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def remove_latex_environments(text: str) -> str:
    """
    Remove LaTeX environments like title, abstract, tables, etc.
    
    Args:
        text: Input text containing LaTeX environments
    
    Returns:
        Text with LaTeX environments replaced with plain text
    """
    text = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\title\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\section\*\{(.*?)\}', r'\1:', text)
    
    # No table replacement here - tables are handled by format_tables function
    
    return text


def format_tables(text: str) -> str:
    """
    Format LaTeX tables into a more readable plain text format with dashed lines
    only at the top and bottom of each table.
    
    Args:
        text: Input text containing LaTeX tables
    
    Returns:
        Text with tables formatted in plain text
    """
    table_pattern = r'\\begin\{tabular\}.*?\\end\{tabular\}'
    tables = re.finditer(table_pattern, text, re.DOTALL)
    
    replacements = []
    for table_match in tables:
        table_content = table_match.group(0)
        original_table = table_content
        
        # Handle nested tabulars
        nested_pattern = r'\\begin\{tabular\}[^}]*\}(.*?)\\end\{tabular\}'
        while re.search(nested_pattern, table_content, re.DOTALL):
            table_content = re.sub(nested_pattern, r'\1', table_content, flags=re.DOTALL)
        
        # Handle multicolumn
        table_content = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}', r'\1', table_content)
        
        table_content = re.sub(r'&', ' | ', table_content)
        table_content = re.sub(r'\\\\', '\n', table_content)
        
        table_content = re.sub(r'\\begin\{tabular\}[^}]*\}', '', table_content)
        table_content = re.sub(r'\\end\{tabular\}', '', table_content)
        table_content = re.sub(r'\\hline', '', table_content)
        
        lines = [line.strip() for line in table_content.strip().split('\n')]
        lines = [line for line in lines if line and not line.startswith('-')]  # Remove empty lines and dashed lines
        
        if not lines:
            replacements.append((original_table, ''))
            continue
            
        dashed_line = '-' * 50
        formatted_table = f"\n{dashed_line}\n" + "\n".join(lines) + f"\n{dashed_line}\n"
        
        replacements.append((original_table, formatted_table))
    
    for original, replacement in replacements:
        text = text.replace(original, replacement)
    
    # Final cleanup to handle any remaining LaTeX table commands
    text = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\begin\{tabular\}[^}]*\}', '', text)
    text = re.sub(r'\\end\{tabular\}', '', text)
    text = re.sub(r'\\hline', '', text)
    
    # Remove any duplicate dashed lines (more than two consecutive dashes)
    text = re.sub(r'(-{50}\n)\s*(-{50}\n)', r'\1', text)
    
    return text


def format_figure_captions(text: str) -> str:
    """
    Format figure and table captions instead of removing them.
    
    Args:
        text: Input text containing figure and table captions
    
    Returns:
        Text with formatted figure and table captions
    """
    text = re.sub(r'Figure (\d+)\.([^\n]*)', r'[Figure \1] \2', text)
    text = re.sub(r'Table (\d+)\.([^\n]*)', r'[Table \1] \2', text)
    
    text = re.sub(r'\(see Figure (\d+)\)', r'(see [Figure \1])', text)
    text = re.sub(r'\(Figure (\d+)\)', r'([Figure \1])', text)
    text = re.sub(r'\(Table (\d+)\)', r'([Table \1])', text)
    
    return text


def clean_math_notation(text: str) -> str:
    """
    Clean or simplify LaTeX mathematical notation.
    
    Args:
        text: Input text containing LaTeX math
    
    Returns:
        Text with simplified math notation
    """
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    text = re.sub(r'_{([^{}]+)}', r'_\1', text)
    text = re.sub(r'\^{([^{}]+)}', r'^\1', text)
    
    text = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', text)
    text = re.sub(r'\\left\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    
    return text


def convert_special_characters(text: str) -> str:
    """
    Convert LaTeX special characters to plain text.
    
    Args:
        text: Input text containing LaTeX special characters
    
    Returns:
        Text with special characters converted to plain text
    """
    text = re.sub(r'\\\\', '\n', text)
    
    replacements = {
        r'\&': '&',
        r'\%': '%',
        r'\$': '$',
        r'\#': '#',
        r'\_': '_',
        r'\{': '{',
        r'\}': '}',
        r'\~': '~',
        r'\^': '^',
    }
    
    for latex, plain in replacements.items():
        text = text.replace(latex, plain)
    
    return text


def combine_empty_sections(text: str) -> str:
    """
    Combine empty sections with the next section header.
    
    Args:
        text: Input text containing possibly empty section headers
    
    Returns:
        Text with empty sections combined with the next section
    """
    # Pattern to find empty sections: \section*{Name1} followed by only whitespace and then \section*{Name2}
    pattern = r'\\section\*\{([^}]*)\}\s*\\section\*\{([^}]*)\}'
    
    replacement = r'\\section*{\1 \2}'
    
    # Keep applying the pattern until no more changes are made
    prev_text = ""
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, replacement, text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text with irregular whitespace
    
    Returns:
        Text with normalized whitespace
    """
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_remaining_latex(text: str) -> str:
    """
    Clean up any remaining LaTeX commands that weren't caught by other functions.
    
    Args:
        text: Input text potentially containing LaTeX commands
    
    Returns:
        Text with remaining LaTeX commands removed or simplified
    """
    # Remove common LaTeX environments and commands that might be missed
    patterns = [
        (r'\\begin\{[^}]*\}', ''),  # Remove \begin{...}
        (r'\\end\{[^}]*\}', ''),    # Remove \end{...}
        (r'\\hline', ''),           # Remove \hline
        (r'\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}', r'\1'),  # Replace \multicolumn{n}{align}{text} with text
        (r'\\textbf\{([^}]*)\}', r'\1'),  # Replace \textbf{text} with text
        (r'\\textit\{([^}]*)\}', r'\1'),  # Replace \textit{text} with text
        (r'\\emph\{([^}]*)\}', r'\1'),    # Replace \emph{text} with text
        (r'\\cite\{[^}]*\}', '[citation]'), # Replace \cite{...} with [citation]
        (r'\\ref\{[^}]*\}', '[ref]'),     # Replace \ref{...} with [ref]
        (r'\\label\{[^}]*\}', ''),        # Remove \label{...}
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
    
    return text


def main():
    """
    Main function to parse command-line arguments and process files.
    """
    parser = argparse.ArgumentParser(description='Process text files with custom parsing functions.')
    parser.add_argument('--input', help='Input file or directory containing text files')
    parser.add_argument('--output', help='Output directory for processed files')
    args = parser.parse_args()
    
    parsing_functions = [
        remove_footnotes,
        remove_author_sections,
        remove_acknowledgment_sections,
        remove_reference_sections,
        clean_math_notation,
        convert_special_characters,
        format_figure_captions,
        combine_empty_sections,
        normalize_whitespace,
        # Consider re-adding clean_remaining_latex if specific issues persist after other cleanups.
        # format_tables and remove_latex_environments are very aggressive and might remove desired content.
        # They are kept here for reference if deep cleaning of specific LaTeX structures is needed.
    ]
    
    process_files(args.input, args.output, parsing_functions)


if __name__ == '__main__':
    main()
