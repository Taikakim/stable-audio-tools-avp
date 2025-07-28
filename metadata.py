# script proven to work it 26.23.2025
# 11.04.2025 added cap words and folder and filename options
import re 
import random
import os

# ANSI color codes for Colab
class Colors:
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_custom_metadata(info, audio):
    """
    Generate a prompt that combines filename processing with PROMPT file contents,
    with added probabilities for capitalized words and prompt prefixes.

    Parameters:
        info: Dictionary containing file information
        audio: Audio data

    Returns:
        Dictionary with the generated prompt
    """
    # Configuration settings
    ENABLE_KEY_DROPPING = True       # Drop an entire key-value pair occasionally
    KEY_DROP_CHANCE = 0.25           # 25% chance to drop a key-value pair when enabled
    ENABLE_WORD_DROPPING = True      # Drop individual words occasionally
    WORD_DROP_CHANCE = 0.05          # 5% chance per word to be dropped
    UPPERCASE_CHANCE = 0.25          # 25% chance for the entire prompt to be uppercase
    DEBUG_PRINT = False              # Print the generated prompt for debugging
    FOLDER_DEPTH = 0                 # How many parent folders to include in the prompt (0 = filename only)
    USE_COLOR = True                 # Use colored output in the console
    CAPS_WORD_CHANCE = 0.15           # 15% chance for individual words to be capitalized

    # Get the absolute and relative paths from the info dictionary
    abs_path = info.get("path", "")
    rel_path = info.get("relpath", "")

    # Extract filename and directory information
    filename = os.path.basename(rel_path)
    base_filename = os.path.splitext(filename)[0]
    directory = os.path.dirname(abs_path)  # Use absolute path for finding the PROMPT file

    # Initialize the prompt prefix
    prompt_prefix = ""

    # Determine the prompt prefix based on probabilities
    prefix_choice = random.random()
    if prefix_choice < 0.60:
        # 60% chance: .PROMPT, filename, and path
        path_without_prefix = rel_path.replace("/content/drive/MyDrive/StableAudioOpen/", "", 1).lstrip("/")
        prompt_prefix = f".PROMPT {base_filename} {path_without_prefix}"
    elif prefix_choice < 0.80:
        # 20% chance: .PROMPT and filename
        prompt_prefix = f".PROMPT {base_filename}"
    elif prefix_choice < 0.95:
        # 15% chance: filename
        prompt_prefix = base_filename
    else:
        # 5% chance: no prompt at all
        prompt_prefix = ""

    # Process the path with the requested folder depth for the main prompt
    if FOLDER_DEPTH > 0:
        # Split the relative path into components
        path_parts = rel_path.split(os.sep)

        # Determine how much of the path to keep
        if FOLDER_DEPTH >= len(path_parts) - 1:
            # If requested depth is more than available, use the full path
            path_with_depth = rel_path
        else:
            # Use last FOLDER_DEPTH folders plus filename
            path_with_depth = os.path.join(*path_parts[-(FOLDER_DEPTH+1):])
    else:
        # FOLDER_DEPTH is 0, just use the filename
        path_with_depth = filename

    # First process the filename using the original rules
    filename_prompt = path_with_depth

    # Remove file extensions (case insensitive)
    filename_prompt = re.sub(r'\.(?:flac|wav|mp3|ogg)$', '', filename_prompt, flags=re.IGNORECASE)

    # Replace /, \, _, and - with whitespace
    filename_prompt = re.sub(r'[/\\_-]', ' ', filename_prompt)

    # Remove any extra whitespace
    filename_prompt = ' '.join(filename_prompt.split())

    # Initialize the main prompt with the processed filename
    prompt = filename_prompt

    # Now check if we have a PROMPT file to append additional information
    # Look for PROMPT file in the same directory as the audio file
    prompt_path = os.path.join(directory, f"{base_filename}.PROMPT")

    # Debug the PROMPT file path
    if DEBUG_PRINT:
        print(f"Looking for PROMPT file at: {prompt_path}")

    # Check case variations if the exact match isn't found
    if not os.path.exists(prompt_path):
        for ext in ['.prompt', '.Prompt']:
            alt_path = os.path.join(directory, f"{base_filename}{ext}")
            if os.path.exists(alt_path):
                prompt_path = alt_path
                break

    if os.path.exists(prompt_path):
        if DEBUG_PRINT:
            print(f"Found PROMPT file: {prompt_path}")
        try:
            with open(prompt_path, 'r') as f:
                content = f.read()

            # Extract all sections enclosed in curly braces
            prompt_sections = []

            # Find all sections like {Key: C minor} and preserve them with braces
            sections = re.findall(r'({[^{}]+})', content)

            for section in sections:
                # Check if this section should be dropped (random chance)
                if ENABLE_KEY_DROPPING and random.random() < KEY_DROP_CHANCE:
                    continue

                # Keep all sections from the PROMPT file
                prompt_sections.append(section)

            # Shuffle the order of prompt sections
            random.shuffle(prompt_sections)

            # Add the PROMPT file sections to the filename-based prompt
            if prompt_sections:
                if prompt:  # If we already have filename content
                    prompt += ", " + " ".join(prompt_sections)
                else:
                    prompt = " ".join(prompt_sections)
        except Exception as e:
            if DEBUG_PRINT:
                print(f"Error reading PROMPT file: {str(e)}")
    elif DEBUG_PRINT:
        print(f"No PROMPT file found for: {filename}")

    # Apply word-level dropout if enabled
    if ENABLE_WORD_DROPPING and prompt:
        # Split by spaces while preserving {} sections
        # This regex finds either a complete {...} section or words between them
        tokens = re.findall(r'({[^{}]+}|\S+)', prompt)
        filtered_tokens = []

        for token in tokens:
            # Don't drop {} sections, only consider dropping individual words
            if token.startswith('{') and token.endswith('}'):
                filtered_tokens.append(token)
            elif random.random() >= WORD_DROP_CHANCE:
                # Keep this word (95% chance)
                filtered_tokens.append(token)
            # Otherwise drop the word (5% chance)

        prompt = " ".join(filtered_tokens)

    # Randomize capitalization of the entire prompt
    if random.random() < UPPERCASE_CHANCE:
        def preserve_braces_upper(text):
            return re.sub(r'({[^{}]+})',
                          lambda m: m.group(0).upper(),
                          text.upper())
        prompt = preserve_braces_upper(prompt)
    else:
        def preserve_braces_lower(text):
            return re.sub(r'({[^{}]+})',
                          lambda m: m.group(0).lower(),
                          text.lower())
        prompt = preserve_braces_lower(prompt)

    # Apply probability for individual word capitalization
    final_prompt_tokens = []
    for word in prompt.split():
        if not (word.startswith('{') and word.endswith('}')) and random.random() < CAPS_WORD_CHANCE:
            final_prompt_tokens.append(word.upper())
        else:
            final_prompt_tokens.append(word)
    prompt = " ".join(final_prompt_tokens)

    # Add the prefix to the beginning of the prompt if it's not empty
    if prompt_prefix:
        prompt = f"{prompt_prefix}, {prompt}".lstrip(", ")

    # Debug printing
    if DEBUG_PRINT:
        if USE_COLOR:
            print(f"Generated prompt: {Colors.YELLOW}{prompt}{Colors.RESET}")
        else:
            print(f"Generated prompt: {prompt}")

    return {"prompt": prompt}