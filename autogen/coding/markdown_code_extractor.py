# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import re

from ..code_utils import CODE_BLOCK_PATTERN, MD_CODE_BLOCK_PATTERN, UNKNOWN, content_str, infer_lang, BASH_CODE_BLOCK_PATTERN
from ..doc_utils import export_module
from ..types import UserMessageImageContentPart, UserMessageTextContentPart
from .base import CodeBlock, CodeExtractor
from ..cmbagent_utils import cmbagent_debug
import json

__all__ = ("MarkdownCodeExtractor",)

# original ag2 code
# @export_module("autogen.coding")
# class MarkdownCodeExtractor(CodeExtractor):
#     """(Experimental) A class that extracts code blocks from a message using Markdown syntax."""

#     def extract_code_blocks(
#         self, message: Union[str, list[Union[UserMessageTextContentPart, UserMessageImageContentPart]], None]
#     ) -> list[CodeBlock]:
#         """(Experimental) Extract code blocks from a message. If no code blocks are found,
#         return an empty list.

#         Args:
#             message (str): The message to extract code blocks from.

#         Returns:
#             List[CodeBlock]: The extracted code blocks or an empty list.
#         """
#         ## cmbagent debug print: 
#         print('in markdown_code_extractor.py extract_code_blocks message: ', message)
#         text = content_str(message)
#         print('in markdown_code_extractor.py extract_code_blocks text: ', text)
#         match = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
#         if not match:
#             return []
#         code_blocks = []
#         for lang, code in match:
#             if lang == "":
#                 lang = infer_lang(code)
#             if lang == UNKNOWN:
#                 lang = ""
#             code_blocks.append(CodeBlock(code=code, language=lang))
#         return code_blocks


# cmbagent modified code
@export_module("autogen.coding")
class MarkdownCodeExtractor(CodeExtractor):
    """(Experimental) A class that extracts code blocks from a message using Markdown syntax,
    and also supports extraction from JSON messages that contain a 'python_code' field.

    Supports different code extraction patterns based on execution policies.
    """

    def __init__(self, execution_policies: dict[str, bool] | None = None):
        """Initialize the MarkdownCodeExtractor.

        Args:
            execution_policies: Optional dictionary mapping language types to execution policy.
                               Used to determine which code block patterns to prioritize.
                               Example: {'bash': True, 'python': False} will prioritize bash extraction.
        """
        self.execution_policies = execution_policies or {}

    def extract_code_blocks(
        self, message: str | list[UserMessageTextContentPart | UserMessageImageContentPart] | None
    ) -> list[CodeBlock]:
        """Extract code blocks from a message. First, if the message is valid JSON and contains
        a "python_code" field, extract that code. Otherwise, fall back to the Markdown regex extraction.

        Args:
            message (str): The message to extract code blocks from.

        Returns:
            List[CodeBlock]: The extracted code blocks or an empty list.
        """
        # cmbagent debug print: 
        # print('in markdown_code_extractor.py extract_code_blocks message: ', message)
        text = content_str(message)
        if cmbagent_debug:
            print('in markdown_code_extractor.py extract_code_blocks text: ', text)

        # Special handling for researcher_response_formatter has been disabled
        # since name parameter was removed to fix compatibility issues
        # if name == "researcher_response_formatter":
        #     if cmbagent_debug:
        #         print('in markdown_code_extractor.py name: ', name)
        #         print('in markdown_code_extractor.py text: ', text)
        #     match = re.findall(MD_CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
        #     if cmbagent_debug:
        #         print('in markdown_code_extractor.py match: ', match)
        #     if not match:
        #         return []
        #     code_blocks = []
        #     for code in match:
        #         code_blocks.append(CodeBlock(code=code, language="markdown"))
        #     return code_blocks

        # Attempt to parse the message as JSON and extract "python_code"
        try:
            data = json.loads(text)
            if "python_code" in data:
                python_code = data["python_code"]
                # Optionally, you could further process python_code here if needed
                return [CodeBlock(code=python_code, language="python")]
            elif "structured_code" in data:  # Added support for structured_code.
                structured_code = data["structured_code"]
                return [CodeBlock(code=structured_code, language="python")]
        except json.JSONDecodeError:
            # The message is not valid JSON; fall back to Markdown extraction.
            ## cmbagent debug print: 
            # print('in markdown_code_extractor.py message is not valid JSON, fall back to Markdown extraction')
            pass

        # First try to extract markdown code blocks
        md_match = re.findall(MD_CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
        if md_match:
            if cmbagent_debug:
                print('in markdown_code_extractor.py found markdown blocks: ', len(md_match))
            code_blocks = []
            for code in md_match:
                code_blocks.append(CodeBlock(code=code, language="markdown"))
            return code_blocks

        # Determine which pattern(s) to try based on execution policies
        patterns_to_try = []

        # Check if bash execution is enabled
        bash_enabled = self.execution_policies.get('bash', False) or \
                      self.execution_policies.get('sh', False) or \
                      self.execution_policies.get('shell', False)

        # Check if python execution is enabled
        python_enabled = self.execution_policies.get('python', False)

        # Prioritize bash pattern if bash is enabled
        if bash_enabled:
            patterns_to_try.append(('bash', BASH_CODE_BLOCK_PATTERN))

        # Add python pattern if python is enabled
        if python_enabled:
            patterns_to_try.append(('python', CODE_BLOCK_PATTERN))

        # If no specific policies are set, try both patterns
        if not patterns_to_try:
            patterns_to_try.append(('bash', BASH_CODE_BLOCK_PATTERN))
            patterns_to_try.append(('python', CODE_BLOCK_PATTERN))

        if cmbagent_debug:
            print(f'in markdown_code_extractor.py trying patterns for: {[p[0] for p in patterns_to_try]}')

        # Try each pattern in order
        for pattern_name, pattern in patterns_to_try:
            match = re.findall(pattern, text, flags=re.DOTALL)
            if cmbagent_debug:
                print(f'in markdown_code_extractor.py {pattern_name} pattern match: {match}')

            if match:
                code_blocks = []
                for lang, code in match:
                    # If lang is empty or None, use the pattern name as fallback
                    if not lang:
                        lang = pattern_name
                    if lang == UNKNOWN:
                        lang = ""
                    code_blocks.append(CodeBlock(code=code, language=lang))
                return code_blocks

        return []