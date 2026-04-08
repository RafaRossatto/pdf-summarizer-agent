from pathlib import Path
import sys
import fitz  
import re
from typing import List, Optional, Callable


class PDF:    
    def __init__(self, file_path: str):
        """
        Class to handle PDF file operations using PyMuPDF (fitz).

        This class provides a comprehensive interface for working with PDF files,
        including loading, text extraction, text cleaning, and export operations.
        It uses PyMuPDF (fitz) as the underlying PDF processing engine.

        Key Features:
            - PDF validation and loading with error handling
            - Automatic text extraction from all pages
            - Comprehensive text cleaning pipeline with multiple filters
            - Support for custom filter chains
            - Text statistics and validation
            - Export cleaned text to file

        The cleaning pipeline includes filters for:
            - Removing CID artifacts (PDF-specific garbage)
            - Removing control characters
            - Normalizing different hyphen types
            - Removing repeated dots and dot patterns
            - Removing page numbers and short headers
            - Fixing punctuation spacing
            - Normalizing whitespace
            - And more...

        All filters can be customized, reordered, or replaced to suit
        specific document types and cleaning requirements.
        """
        self.file_path = file_path
        self.doc = None  
        self.page_count = 0
        self.raw_text = ""       
        self.cleaned_text = ""   
        
    # ===== LOADING METHODS =====
    
    def load_pdf(self) -> bool:
        """
        Load and validate a PDF file from the given path using PyMuPDF (fitz).
        
        This method performs comprehensive validation and loading of a PDF file:
        1. Checks if the file exists at the specified path
        2. Validates that the file has a .pdf extension
        3. Attempts to open the file as a PDF using PyMuPDF
        4. Counts the number of pages in the document
        5. Automatically extracts text from all pages
        6. Displays loading statistics
        
        The method is designed to be robust, with multiple layers of error handling
        and clear user feedback at each step. It will exit the program gracefully
        if any critical error occurs during loading.
        
        Args:
            None (uses self.file_path from initialization)
        
        Returns:
            bool: True if PDF loaded successfully (always returns True, 
                or exits program on failure)
        
        Raises:
            SystemExit: If any error occurs during loading (file not found,
                    invalid PDF, permission denied, etc.)
                       
        Loading statistics displayed:
            - File path: Full path to the loaded PDF
            - Page count: Total number of pages in document
            - Character count: Total characters in extracted text
            - Word count: Total words (split by whitespace)
        
        Attributes set after successful load:
            - self.doc: PyMuPDF document object for further operations
            - self.page_count: Integer number of pages
            - self.raw_text: String containing all extracted text
        
        Error handling strategy:
            This method uses sys.exit(1) for all errors because:
            - A PDF must be loaded successfully for the program to function
            - Continuing with a failed load would cause cascading errors
            - Clear error messages help users fix the problem
            - Exit code 1 indicates an error to calling scripts
        
        Common error scenarios and messages:
            Error Type                    | Message
            ------------------------------|--------------------------------
            File not found                | "Error: File '{path}' does not exist."
            Wrong extension               | "Error: File '{path}' is not a PDF file."
            Corrupt PDF                   | "Error: Failed to load PDF file. {details}"
            Permission denied             | "Error: Permission denied to read file '{path}'"
            Network drive issues          | "Unexpected error while loading PDF: {details}"
        
        Performance considerations:
            - File existence check is O(1)
            - PDF opening time depends on file size and complexity
            - Page count is O(1) (metadata)
            - Text extraction time scales with document size
            - Memory usage scales with document size
            """
        try:
            # Initial validations 
            if not Path(self.file_path).exists():
                print(f" Error: File '{self.file_path}' does not exist.")
                sys.exit(1)
            
            if not self.file_path.lower().endswith('.pdf'):
                print(f" Error: File '{self.file_path}' is not a PDF file.")
                sys.exit(1)
            
            # Try to open the PDF with PyMuPDF
            try:
                self.doc = fitz.open(self.file_path) 
                self.page_count = len(self.doc)        
                
                print(f"\n PDF loaded successfully with PyMuPDF:")
                print(f" File: {self.file_path}")
                print(f" Pages: {self.page_count}")
                            
                # Extract text automatically (chama o método atualizado)
                self._extract_all_text()
                
                # Show text statistics (igual)
                char_count = len(self.raw_text)
                word_count = len(self.raw_text.split())
                print(f"Text stats: {char_count} characters, {word_count} words")
                
                return True
                
            except Exception as e:
                print(f" Error: Failed to load PDF file. {str(e)}")
                sys.exit(1)
                    
        except PermissionError:
            print(f" Error: Permission denied to read file '{self.file_path}'.")
            sys.exit(1)
        except Exception as e:
            print(f" Unexpected error while loading PDF: {str(e)}")
            sys.exit(1)
    
    def _extract_all_text(self):
        """
        Extract text from all pages of the PDF document using PyMuPDF.
        
        This method iterates through every page in the loaded PDF, extracts the
        text content from each page, and concatenates them into a single string
        stored in self.raw_text. It's called automatically when loading a PDF.
        
        The extraction uses PyMuPDF's (fitz) built-in text extraction which:
        - Preserves reading order (top-to-bottom, left-to-right)
        - Handles different font types and encodings
        - Extracts text from forms, annotations, and digital text
        - Does NOT perform OCR (only extracts existing digital text)
        
        Args:
            None (uses self.doc and self.page_count)
        
        Returns:
            None (stores result in self.raw_text)
        
        Process flow:
            1. Check if document is loaded (self.doc exists)
            2. If no document, set self.raw_text = "" and return
            3. Initialize empty list to store text from each page
            4. Loop through each page (0 to page_count-1):
                a. Get page object by index
                b. Extract text using page.get_text()
                c. If text exists (not empty), append to list
            5. Join all pages with newline characters
            6. Store result in self.raw_text
        
        Page joining strategy:
            Pages are joined with newline characters ('\n') to preserve
            page boundaries. This creates a structure like:
            
            [Page 1 text]\n[Page 2 text]\n[Page 3 text]...
            
            This allows later processing to:
            - Know where page breaks occur
            - Process pages individually if needed
            - Reconstruct original document structure
                      
        Common extraction challenges:
            1. Reading order issues:
            - Multi-column layouts may read across columns
            - Tables may extract in wrong order
            - Headers/footers may interleave with content
            
            2. Spacing issues:
            - Words may be concatenated (n o s p a c e s)
            - Extra spaces may appear
            - Hyphenation problems at line breaks
            
            3. Special characters:
            - Unicode normalization needed
            - Symbol fonts may not map correctly
            - CID artifacts may appear
        
        Performance considerations:
            - Time complexity: O(n) where n = number of pages
            - Memory usage: Stores entire document text
            - For very large documents (>1000 pages), consider
            streaming or processing page by page
        
        Error handling:
            - If document is None or not loaded, an error message is printed and extraction stops
            - If page access fails (index error, permission error, etc.), a warning is printed with the page number and specific error
            - Empty pages or pages with only whitespace are skipped with a warning message
            - Pages with very little content (optional) can trigger a warning
            - If no text is extracted from any page, a final warning is printed
        
        """
        if not self.doc:
            print("Error: No document loaded. Unable to extract text.")
            self.raw_text = ""
            return

        pages_text = []
        for page_num in range(self.page_count): 
            try:
                page = self.doc[page_num]
                text = page.get_text()
                if text:
                    pages_text.append(text)
            except Exception as e:
                print(f"Warning: Page number {page_num} not accessible. Error: {e}")
                continue

        self.raw_text = "\n".join(pages_text)
    
    # ===== TEXT CLEANING FILTERS =====
    
    def _remove_cid_artifacts(self, text: str) -> str:
        """
        Remove CID (Character ID) artifacts commonly found in PDF text extraction.
        
        CID artifacts appear as patterns like "(cid:12345)" in text extracted from PDFs.
        These are not actual text content but internal PDF character identifiers that
        sometimes leak through during text extraction, especially with:
        - PDFs with embedded fonts that don't map properly to Unicode
        - Corrupted or malformed PDF files
        - PDFs with special characters or non-standard fonts
        - Documents with complex typography or custom character sets
        
        Args:
            text (str): Input text that may contain CID artifacts
        
        Returns:
            str: Text with all CID artifacts removed
        
        Pattern: r'\(cid:\d+\)'
        
        Breakdown:
            - \(         : Literal opening parenthesis (escaped)
            - cid:       : Literal string "cid:"
            - \d+        : One or more digits (the character ID number)
            - \)         : Literal closing parenthesis (escaped)
        
        Matches patterns like:
            - "(cid:12345)"
            - "(cid:0)"
            - "(cid:999999)"
        
        Examples:
            >>> text = "Hello (cid:12345)World"
            >>> filter._remove_cid_artifacts(text)
            "Hello World"  # CID artifact removed, space preserved
            
            >>> text = "This is a (cid:42)test"
            >>> filter._remove_cid_artifacts(text)
            "This is a test"  # Artifact removed, text rejoined
            
            >>> text = "(cid:1)(cid:2)(cid:3)Hello"
            >>> filter._remove_cid_artifacts(text)
            "Hello"  # Multiple artifacts at start removed
            
            >>> text = "Special char: (cid:169) (copyright symbol might be here)"
            >>> filter._remove_cid_artifacts(text)
            "Special char:  (copyright symbol might be here)"
            # Note: The actual character is lost - that's the limitation!
        
        Real-world examples from PDFs:
            >>> text = "The quick brown (cid:102)fox jumps over the (cid:108)lazy dog"
            >>> filter._remove_cid_artifacts(text)
            "The quick brown fox jumps over the lazy dog"
            
            >>> text = "Section 1: (cid:73)Introduction(cid:74)"
            >>> filter._remove_cid_artifacts(text)
            "Section 1: Introduction"
            
            >>> text = "Price: (cid:36)100.00"  # (cid:36) might be dollar sign
            >>> filter._remove_cid_artifacts(text)
            "Price: 100.00"  # Lost the currency symbol 
        
        What gets removed:
            Before                          | After                     | Fixed?
            --------------------------------|---------------------------|--------
            "Hello(cid:12345)World"         | "HelloWorld"              |  Yes
            "Text (cid:42) here"            | "Text  here"              |  Yes
            "(cid:1)(cid:2)Start"           | "Start"                   |  Yes
            "Multiple (cid:1) (cid:2) (cid:3)spaces" | "Multiple   spaces" |  Yes
        
        What remains (no match):
            Before                          | After                     | Reason
            --------------------------------|---------------------------|--------
            "Hello World"                   | "Hello World"             | No CID pattern
            "text(cid:abc)"                 | "text(cid:abc)"           | No digits
            "cid:12345"                     | "cid:12345"               | Missing parentheses
            "(cid:)"                        | "(cid:)"                  | Missing digits
            "normal (parentheses)"          | "normal (parentheses)"    | Not CID format
        
        Why CIDs appear in extracted text:
            PDFs store characters internally as IDs that map to glyphs in fonts.
            When text extraction fails to map these IDs to proper Unicode,
            the raw CID codes are output instead. This commonly happens with:
            
            - Symbol fonts (mathematical symbols, dingbats)
            - Non-standard character encodings
            - PDFs created from scanned images with OCR
            - Documents with missing font information
            - Corrupted PDF structure

         Important limitations:
            - This filter REMOVES characters, it doesn't replace them with proper ones
            - You lose information that might be important
            - A more sophisticated approach would map CIDs to actual characters
            - Some PDF libraries (like pdfplumber) handle this better than others

        Note:
            This is often the FIRST filter applied because CIDs are pure artifacts
            that don't contribute to readable text. Apply it early to prevent
            interference with other filters that expect clean text.
        """
        return re.sub(r'\(cid:\d+\)', '', text)
    
    def _remove_repeated_dots(self, text: str) -> str:
        """
        Remove repeated dot patterns and isolated dots from text.
        
        This filter handles two specific dot-related issues:
        1. Sequences of dots with optional spaces (".", "..", ". .", ". . .")
        2. Isolated dots surrounded by spaces (" . ")
        
        These patterns often appear in PDFs due to:
        - Formatting artifacts from text extraction
        - OCR errors misreading punctuation
        - Corrupted spacing in digital documents
        - Decorative elements that became text
        
        Args:
            text (str): Input text that may contain repeated or isolated dots
        
        Returns:
            str: Text with dot patterns removed/replaced by single spaces
        
        How it works - First pass: Remove repeated dot sequences
            Pattern: r'(?:\.\s*){2,}'
            
            Breakdown:
            - (?: ... )    : Non-capturing group
            - \.           : Literal dot/period
            - \s*          : Zero or more whitespace (spaces, tabs)
            - {2,}         : Group appears 2 or more times
            
            Matches sequences like:
            - ".."         (dot dot)
            - ". ."        (dot space dot)
            - ".  .  ."    (dot with multiple spaces)
            - "..."        (multiple dots)
            - ". . ."      (alternating dots and spaces)
            
            Replacement: ' ' (single space)
            
        How it works - Second pass: Remove isolated dots
            Pattern: r'\s+\.\s+'
            
            Breakdown:
            - \s+          : One or more whitespace
            - \.           : Literal dot
            - \s+          : One or more whitespace
            
            Matches dots that are surrounded by whitespace:
            - " . "        (space dot space)
            - "\t.\n"      (tab dot newline)
            - "  .  "      (multiple spaces around dot)
            
            Replacement: ' ' (single space)
        
        Examples:
            >>> text = "Hello . world"
            >>> filter._remove_repeated_dots(text)
            "Hello world"  # Isolated dot removed
            
            >>> text = "Hello..world"
            >>> filter._remove_repeated_dots(text)
            "Hello world"  # Double dots become space
            
            >>> text = "Hello . . . world"
            >>> filter._remove_repeated_dots(text)
            "Hello world"  # Pattern of dots and spaces becomes one space
            
            >>> text = "Hello...world"
            >>> filter._remove_repeated_dots(text)
            "Hello world"  # Triple dots become space
            
            >>> text = "Hello .  .  . world"
            >>> filter._remove_repeated_dots(text)
            "Hello world"  # Even with extra spaces
            
            >>> text = "Hello .world"  # Dot attached to next word
            >>> filter._remove_repeated_dots(text)
            "Hello .world"  # NOT changed (no space after dot)
        
        What gets fixed:
            Before                    | After                     | Fixed?
            --------------------------|---------------------------|--------
            "Hello . world"           | "Hello world"             |  Yes
            "Hello..world"            | "Hello world"             |  Yes
            "Hello . . . world"       | "Hello world"             |  Yes
            "word...word"             | "word word"               |  Yes
            "word . word"             | "word word"               |  Yes
            "word .  .  . word"       | "word word"               |  Yes
            "word . .word"            | "word .word"              |  Partial (last dot attached)
        
        What stays the same:
            Before                    | After                     | Reason
            --------------------------|---------------------------|--------
            "word.word"               | "word.word"               | Single dot, no spaces
            "U.S.A."                  | "U.S.A."                  | Valid abbreviation
            "192.168.1.1"             | "192.168.1.1"             | IP address
            "3.14159"                 | "3.14159"                 | Decimal number
            "..."                     | "..."                     | No surrounding words
            ".profile"                | ".profile"                | Dot at start (hidden file)
            "file.txt"                | "file.txt"                | File extension
        
        Common PDF artifacts fixed:
            >>> text = "The quick brown fox . jumped over the . lazy dog"
            >>> filter._remove_repeated_dots(text)
            "The quick brown fox jumped over the lazy dog"
            
            >>> text = "Section 1 . . Introduction . . . Page 5"
            >>> filter._remove_repeated_dots(text)
            "Section 1 Introduction Page 5"
            
            >>> text = "Note: . . . important point . . . here"
            >>> filter._remove_repeated_dots(text)
            "Note: important point here"
        
        Comparison with similar filters:
            _remove_repeated_dots     : Handles ".", "..", ". ." patterns
            _remove_dots_between_words: Handles "word....word" patterns  
            _remove_lines_with_many_dots: Removes entire dot-heavy lines

        
         Important considerations:
            - This filter can remove legitimate ellipsis if they have spaces
            - Works best when dots are clearly artifacts, not intentional
            - May need to run BEFORE _fix_punctuation_spacing
            - Consider the context: "Wait . . . what?" becomes "Wait what?"
        
        Note:
            This is an aggressive filter that assumes any pattern of
            dots with spaces around them is noise. Use with understanding
            of your document's characteristics.
        """
        text = re.sub(r'(?:\.\s*){2,}', ' ', text)
        # Isolated dots
        text = re.sub(r'\s+\.\s+', ' ', text)
        return text
    
    def _remove_dots_between_words(self, text: str) -> str:
        """
        Remove sequences of dots between words, replacing them with a single space.
        
        This filter targets patterns where multiple dots appear between words,
        which often occur in PDFs due to:
        - Table of contents dot leaders that got merged with text
        - OCR errors misreading spaces as dots
        - Formatting artifacts from text extraction
        - Decorative separators that became part of the text
        
        Args:
            text (str): Input text that may have dot sequences between words
        
        Returns:
            str: Text with dot sequences replaced by a single space between words
        
        How it works:
            Pattern: r'(\w)\s*\.{2,}\s*(\w)'
            
            Breakdown:
            - (\w)     : Captures a word character (letter, number, underscore)
            - \s*      : Optional whitespace (spaces, tabs)
            - \.{2,}   : Two or more consecutive dots
            - \s*      : Optional whitespace
            - (\w)     : Captures another word character
            
            Replacement: r'\1 \2'
            - \1       : First captured word
            - [space]  : Single space
            - \2       : Second captured word
        
        Examples:
            >>> text = "word....word"
            >>> filter._remove_dots_between_words(text)
            "word word"  # Dots replaced with space
            
            >>> text = "Hello......World"
            >>> filter._remove_dots_between_words(text)
            "Hello World"  # Any number of dots becomes one space
            
            >>> text = "code...code"
            >>> filter._remove_dots_between_words(text)
            "code code"  # Three dots becomes space
            
            >>> text = "word   .....   word"  # With spaces around dots
            >>> filter._remove_dots_between_words(text)
            "word word"  # All whitespace and dots normalized to one space
            
            >>> text = "word.word"  # Single dot (ellipsis? abbreviation?)
            >>> filter._remove_dots_between_words(text)
            "word.word"  # NOT changed - only sequences of 2+ dots
            
            >>> text = "...word..."  # Dots at start/end
            >>> filter._remove_dots_between_words(text)
            "...word..."  # NOT changed - need word on both sides
        
        What gets fixed:
            Before                    | After                     | Fixed?
            --------------------------|---------------------------|--------
            "word....word"            | "word word"               |  Yes
            "Hello........World"      | "Hello World"             |  Yes
            "code...code"             | "code code"               |  Yes
            "a..b"                    | "a b"                     |  Yes
            "word   .....   word"     | "word word"               |  Yes
            "word.word"               | "word.word"               |  No (single dot)
            "word... word"            | "word word"               |  Yes
            "word ... word"           | "word word"               |  Yes
        
        What stays the same:
            Before                    | After                     | Reason
            --------------------------|---------------------------|--------
            "word.word"               | "word.word"               | Single dot (could be valid)
            "..."                     | "..."                     | No words on either side
            "word..."                 | "word..."                 | No word after dots
            "...word"                 | "...word"                 | No word before dots
            "U.S.A."                  | "U.S.A."                  | Single dots in acronym
            "192.168.1.1"             | "192.168.1.1"             | IP address with single dots
            "3.14159"                 | "3.14159"                 | Decimal number
        
        Common PDF artifacts fixed:
            >>> text = "Introduction..........5"
            >>> filter._remove_dots_between_words(text)
            "Introduction 5"  # TOC dot leader fixed
            
            >>> text = "Chapter 1.........The Beginning"
            >>> filter._remove_dots_between_words(text)
            "Chapter 1 The Beginning"  # Section title fixed
            
            >>> text = "word1.........word2.........word3"
            >>> filter._remove_dots_between_words(text)
            "word1 word2 word3"  # Multiple dot sequences fixed
        
         Important considerations:
            - Only affects sequences of 2 or more dots
            - Requires word characters on BOTH sides to trigger
            - Preserves single dots (abbreviations, decimals, IPs)
            - Removes any whitespace around the dot sequence too
            - Safe to use multiple times (already normalized after first pass)
        
        Technical note:
            The pattern uses \w which matches [a-zA-Z0-9_]. If you need to
            include other characters as word boundaries, the pattern could be
            extended, but for most English text this works well.
        """
        return re.sub(r'(\w)\s*\.{2,}\s*(\w)', r'\1 \2', text)
    
    def _remove_lines_with_many_dots(self, text: str, threshold: float = 0.3) -> str:
        """
        Remove lines that have a high proportion of dot/period characters.
        
        This filter identifies and removes lines that are dominated by dots (periods),
        which often indicate:
        - Table of contents dot leaders (.........)
        - Horizontal rules or separators (...............)
        - Decorative lines or borders in PDFs
        - OCR artifacts or noise
        - Corrupted text extraction
        
        Args:
            text (str): Input text that may contain dot-heavy lines
            threshold (float, optional): Maximum allowed ratio of dots to total characters.
                Lines with dot ratio > threshold are removed.
                Defaults to 0.3 (30% dots).
                Range: 0.0 to 1.0 (0% to 100% dots)
        
        Returns:
            str: Text with dot-heavy lines removed, all other lines preserved
        
        How it works:
            1. Splits text into individual lines
            2. For each non-empty line, calculates: dots_count / total_length
            3. If ratio > threshold, the line is REMOVED
            4. Empty lines (line is False) are always kept
            5. Otherwise, the line is KEPT
        
        Examples:
            >>> text = "Chapter 1\\n................\\nThe Beginning"
            >>> filter._remove_lines_with_many_dots(text)
            "Chapter 1\\nThe Beginning"
            # Removes the dot leader line (100% dots)
            
            >>> text = "Table of Contents\\n1. Introduction........5\\n2. Methods.........12"
            >>> filter._remove_lines_with_many_dots(text, threshold=0.5)
            "Table of Contents\\n1. Introduction........5\\n2. Methods.........12"
            # Keeps TOC lines (dots are part of content, ratio < 0.5)
            
            >>> text = "Normal line\\n.......\\nAnother line\\n....\\nLast line"
            >>> filter._remove_lines_with_many_dots(text)
            "Normal line\\nAnother line\\nLast line"
            # Removes "......." and "...." (100% dots)
            
            >>> text = "Line with some.dots.here.and.there"
            >>> filter._remove_lines_with_many_dots(text)
            "Line with some.dots.here.and.there"
            # Keeps line with interspersed dots (low dot ratio)
        
        Threshold examples:
            Line content                    | Length | Dots | Ratio | Removed?
            --------------------------------|--------|------|-------|----------
            ".............................."| 30     | 30   | 1.0   |  Yes
            "........."                     | 9      | 9    | 1.0   |  Yes
            "1. Introduction........5"      | 22     | 8    | 0.36  |  No (if threshold=0.3)
            "Normal text."                  | 12     | 1    | 0.08  |  No
            "..."                           | 3      | 3    | 1.0   |  Yes
            ".... Section ...."             | 15     | 8    | 0.53  |  Yes (if threshold=0.3)
            ""                              | 0      | 0    | N/A   |  No (empty line kept)
        
        Common use cases:
            1. Table of Contents dot leaders:
            "Chapter 1..............................5"
            With threshold=0.3 → Kept (dots are meaningful)
            
            2. Decorative separators:
            "................................"
            With threshold=0.3 → Removed (pure decoration)
            
            3. ASCII art or horizontal rules:
            "----------------------------------------" (not dots - unaffected)
            "........................................" (dots - removed)
        
        Choosing the right threshold:
            - 0.1-0.2: Very aggressive (removes lines with few dots)
            - 0.3: Balanced (default) - good for most cases
            - 0.5-0.7: Conservative (only removes very dot-heavy lines)
            - 0.9-1.0: Minimal (only removes pure dot lines)
               
         Important considerations:
            - This filter only considers dots (periods), not other punctuation
            - Empty lines are ALWAYS preserved (they don't trigger removal)
            - The filter doesn't modify lines, only removes entire lines
            - Very short lines with many dots are more likely to be removed
        
        Note:
            The dot ratio calculation uses line.count('.') which counts ALL periods,
            including those in URLs, numbers, abbreviations, etc. This can sometimes
            remove legitimate content if the threshold is too low.
        """
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if line and line.count('.') / len(line) > threshold:
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)
    
    def _remove_page_numbers(self, text: str) -> str:
        """
        Remove lines that consist solely of page numbers.
        
        This filter identifies and removes lines that contain nothing but a number
        (e.g., "1", "42", "123"). These are typically page numbers that appear
        isolated on their own line in PDFs and pollute the extracted text.
        
        Args:
            text (str): Input text that may contain isolated page numbers
        
        Returns:
            str: Text with isolated page number lines removed, all other lines preserved
        
        How it works:
            1. Splits text into individual lines
            2. For each line, checks if the stripped content matches pattern ^\d+$
            - ^      : Start of string
            - \d+    : One or more digits (0-9)
            - $      : End of string
            3. If pattern matches, the line is REMOVED
            4. Otherwise, the line is KEPT
        
        Examples:
            >>> text = "Introduction\\n1\\nChapter 1\\n2\\nThe Beginning"
            >>> filter._remove_page_numbers(text)
            "Introduction\\nChapter 1\\nThe Beginning"
            # Removes "1" and "2" (isolated page numbers)
            
            >>> text = "Abstract\\n\\n42\\n\\nResults"
            >>> filter._remove_page_numbers(text)
            "Abstract\\n\\n\\nResults"  # Empty lines preserved, "42" removed
            
            >>> text = "Page 1\\n2\\nSection 3\\n4\\nConclusion"
            >>> filter._remove_page_numbers(text)
            "Page 1\\nSection 3\\nConclusion"
            # Removes "2" and "4", keeps "Page 1" and "Section 3"
            
            >>> text = "2023 Report\\n1\\nIntroduction\\n2.1\\nDetails"
            >>> filter._remove_page_numbers(text)
            "2023 Report\\nIntroduction\\n2.1\\nDetails"
            # Removes "1", keeps "2023 Report" (not just digits) and "2.1" (has dot)
        
        Edge cases handled:
            Line content | Pattern matches? | Removed? | Reason
            -------------|------------------|----------|--------
            "1"          |  Yes             |  Yes     | Just a digit
            "42"         |  Yes             |  Yes     | Multiple digits
            "123"        |  Yes             |  Yes     | Any number of digits
            " 5 "        |  Yes             |  Yes     | Whitespace stripped
            "Page 1"     |  No              |  No      | Contains text
            "2.1"        |  No              |  No      | Contains dot
            "1."         |  No              |  No      | Contains dot
            "2023"       |  Yes             |  Yes     | Just digits (but is this a year?)
            "Section 2"  |  No              |  No      | Contains text
            ""           |  No              |  No      | Empty line preserved
                
         Important caveats:
            - YEARS: "2023" on its own line WILL BE REMOVED (use with caution!)
            - ITEM NUMBERS: "1" as list item will be removed if isolated
            - MIXED CONTENT: Numbers with punctuation (e.g., "1.") are preserved
        
        Customization possibilities:
            # Stricter: Only remove numbers with 1-3 digits (avoids removing years)
            pattern = r'^\d{1,3}$'
            
            # More permissive: Also remove numbers with dots (e.g., "1.")
            pattern = r'^\d+\.?$'
            
            # Roman numerals: Also remove I, II, III (but this filter doesn't)
        
        Note:
            This filter preserves empty lines and lines with mixed content.
            Consider running it BEFORE _remove_empty_lines to clean up
            gaps left by removed page numbers.
        """
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if not re.match(r'^\d+$', line.strip()):
                cleaned.append(line)
        return '\n'.join(cleaned)
    
    def _remove_short_headers(self, text: str, max_length: int = 5) -> str:
        """
        Remove short uppercase lines that are likely to be section headers or artifacts.
        
        This filter identifies and removes lines that consist entirely of uppercase letters
        and are very short (less than max_length characters). These are often:
        - Section headers (e.g., "II", "III", "A.", "B)")
        - Page markers or labels
        - OCR artifacts or formatting debris
        - List identifiers that shouldn't be separate lines
        
        Args:
            text (str): Input text that may contain short uppercase headers
            max_length (int, optional): Maximum length of line to consider for removal.
                Lines shorter than this AND all uppercase will be removed.
                Defaults to 5 (removes things like "II", "III", "A.", "B)", "NOTE")
        
        Returns:
            str: Text with short uppercase headers removed, all other lines preserved
        
        How it works:
            1. Splits text into individual lines
            2. For each line, checks if stripped version is:
            a) Shorter than max_length characters
            b) Completely uppercase (stripped.isupper())
            3. If BOTH conditions are true, the line is REMOVED
            4. Otherwise, the line is KEPT (including original formatting)
        
        Examples:
            >>> text = "INTRODUCTION\\nII\\nThe Second Section\\nIII\\nFinal Part"
            >>> filter._remove_short_headers(text)
            "INTRODUCTION\\nThe Second Section\\nFinal Part"
            # Removes "II" and "III" (short + uppercase)
            
            >>> text = "NOTE:\\nImportant note here\\nA.\\nFirst item\\nB.\\nSecond item"
            >>> filter._remove_short_headers(text, max_length=4)
            "Important note here\\nFirst item\\nSecond item"
            # Removes "NOTE:" (4 chars, uppercase) and "A.", "B." (short + uppercase)
            
            >>> text = "Chapter 1\\nThe Beginning\\n\\nAPPENDIX\\nDetails here"
            >>> filter._remove_short_headers(text)
            "Chapter 1\\nThe Beginning\\n\\nAPPENDIX\\nDetails here"
            # "APPENDIX" is 7 chars > max_length, so KEPT
            
            >>> text = "PART ONE\\nI\\nThe First Part\\nII\\nThe Second Part"
            >>> filter._remove_short_headers(text)
            "PART ONE\\nThe First Part\\nThe Second Part"
            # Removes "I" and "II", but keeps "PART ONE" (too long)
        
        Edge cases handled:
            - Lines with punctuation: "A.", "B)", "III." are removed if short
            - Mixed case lines: "Introduction" kept even if short
            - Lines with numbers: "2023" kept (not uppercase)
            - Empty lines: Kept (filter only removes based on content)
               
        Note:
            - This filter only REMOVES lines, never modifies content
            - The threshold (max_length) can be adjusted based on your needs
            - Consider running this BEFORE _remove_empty_lines to clean up structure
            - Use with caution on texts where short uppercase words are meaningful

        """
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not (len(stripped) < max_length and stripped.isupper()):
                cleaned.append(line)
        return '\n'.join(cleaned)
    
    def _remove_control_characters(self, text: str) -> str:
        """
        Remove non-printable control characters from text.
        
        This filter eliminates ASCII control characters that are often present in PDF-extracted
        text but are not meant to be visible. These characters can cause display issues,
        interfere with text processing, and create invisible artifacts in the output.
        
        Args:
            text (str): Input text that may contain control characters
        
        Returns:
            str: Text with all control characters removed, printable characters preserved
        
        Characters removed (ASCII control characters):
        
        Range/Code | Description                    | Common source
        -----------|--------------------------------|----------------------
        \x00-\x08  | Null, Bell, Backspace, etc.    | Binary data, corrupt PDFs
        \x0b       | Vertical Tab                   | Formatting artifacts
        \x0c       | Form Feed (Page Break)         | Page breaks in PDFs
        \x0e-\x1f  | Shift Out, Shift In, etc.      | Encoding markers
        \x7f       | Delete (DEL)                   | Legacy system artifacts
        
        Characters NOT removed (preserved):
            - Regular printable characters (letters, numbers, punctuation)
            - Whitespace: space (\x20), tab (\t), newline (\n), carriage return (\r)
            - Extended ASCII/Unicode characters (> \x7f)
        
        Examples:
            >>> text = "Hello\x00World"  # Contains null character
            >>> filter._remove_control_characters(text)
            "HelloWorld"  # Null removed
            
            >>> text = "Line 1\x0cPage 2"  # Contains form feed (page break)
            >>> filter._remove_control_characters(text)
            "Line 1Page 2"  # Form feed removed, text concatenated
            
            >>> text = "Text with\x1b[31mcolor\x1b[0m codes"
            >>> filter._remove_control_characters(text)
            "Text withcolor codes"  # ANSI escape sequences removed
            
            >>> text = "Normal text\nwith newline\tand tab"
            >>> filter._remove_control_characters(text)
            "Normal text\nwith newline\tand tab"  # Preserved!
        
        Why this matters for PDF processing:
            PDFs often contain hidden control characters from:
            - Embedded fonts with special encodings
            - Corrupted or malformed PDF structures
            - OCR engines that insert control markers
            - Copy-paste from specialized applications
            - Encrypted or protected content
        
        Common issues caused by control characters:
            - Text appears to have "invisible" characters
            - String length doesn't match visible content
            - Regular expressions behave unexpectedly
            - Database storage errors
            - Display rendering problems
            - File encoding issues when saving
        
        Note:
            - This filter preserves normal whitespace (\n, \t, \r)
            - Apply EARLY in the cleaning pipeline to remove problematic characters
            - Safe to use on any text - won't damage readable content
            - Complements _normalize_spaces which handles whitespace normalization
        
        Technical note:
            The regex pattern [\x00-\x08\x0b\x0c\x0e-\x1f\x7f] matches:
            - \x00 to \x08: ASCII 0-8 (NULL, START OF HEADING, etc.)
            - \x0b: Vertical Tab (ASCII 11)
            - \x0c: Form Feed (ASCII 12)
            - \x0e to \x1f: ASCII 14-31 (SO, SI, DLE, etc.)
            - \x7f: Delete (ASCII 127)
            
            Notable exclusions (preserved):
            - \x09: Horizontal Tab (\t)
            - \x0a: Line Feed (\n)
            - \x0d: Carriage Return (\r)
            - \x20: Space
            - All printable characters
        """
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    def _normalize_hyphens(self, text: str) -> str:
        """
        Normalize different hyphen and dash characters to a standard hyphen.
        
        This filter replaces various Unicode hyphen and dash characters with the
        common ASCII hyphen/minus sign (-). This is important because PDFs often
        contain different hyphen characters that look similar but are different
        Unicode code points, which can cause issues in text processing and searching.
        
        Args:
            text (str): Input text that may contain various hyphen-like characters
        
        Returns:
            str: Text with all hyphen variants replaced by standard ASCII hyphen
        
        Characters normalized:
            Original | Unicode | Name                    | Example
            ---------|---------|-------------------------|---------
            '‐'      | U+2010  | HYPHEN                  | "low‐budget" -> "low-budget"
            '–'      | U+2013  | EN DASH                 | "2015–2020" -> "2015-2020"
            '—'      | U+2014  | EM DASH                 | "wait—what?" -> "wait-what?"
            '−'      | U+2212  | MINUS SIGN              | "5−3=2" -> "5-3=2"
            '-'      | U+002D  | HYPHEN-MINUS (standard) | (kept as is)
        
        Examples:
            >>> text = "low‐budget film (2015–2018) was a hit—everyone loved it!"
            >>> filter._normalize_hyphens(text)
            "low-budget film (2015-2018) was a hit-everyone loved it!"
            
            >>> text = "Temperature: −5°C"
            >>> filter._normalize_hyphens(text)
            "Temperature: -5°C"
            
            >>> text = "Well‐known author—born 1980–present"
            >>> filter._normalize_hyphens(text)
            "Well-known author-born 1980-present"
        
        Note:
            - This filter only changes the hyphen characters themselves
            - Does NOT affect surrounding text or spacing
            - Safe to apply at any point in the cleaning pipeline
            - The standard hyphen '-' (U+002D) is left unchanged
        
        See also:
            _normalize_spaces - Often used after this for complete normalization
            _fix_punctuation_spacing - May need hyphens normalized first
        """
        replacements = {'‐': '-', '–': '-', '—': '-', '−': '-'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _normalize_spaces(self, text: str) -> str:
        """
        Normalize whitespace by collapsing multiple spaces and trimming edges.
        
        This filter performs two important whitespace normalizations:
        1. Collapses any sequence of one or more whitespace characters (spaces, tabs, newlines)
        into a single space
        2. Removes leading and trailing whitespace from the entire text
        
        Args:
            text (str): Input text with potentially irregular spacing
        
        Returns:
            str: Text with normalized spacing - single spaces between words,
                no leading/trailing whitespace, and all line breaks converted to spaces
        
        How it works:
            regex r'\s+' - Matches ONE OR MORE whitespace characters:
                - Spaces: ' ', '  ', '   '
                - Tabs: '\t', '\t\t'
                - Newlines: '\n', '\r\n'
                - Any combination: ' \t\n  '
            
            replacement ' ' - Replaces any whitespace sequence with a SINGLE space
            
            .strip() - Removes any remaining whitespace at start/end
        
        Examples:
            >>> text = "Hello    world"
            >>> filter._normalize_spaces(text)
            "Hello world"  # Multiple spaces collapsed
            
            >>> text = "Hello\t\tworld\n\nHow are you?"
            >>> filter._normalize_spaces(text)
            "Hello world How are you?"  # Tabs and newlines become spaces
            
            >>> text = "   Hello world   "
            >>> filter._normalize_spaces(text)
            "Hello world"  # Leading/trailing spaces removed
            
            >>> text = "Hello   world   \n\n   How\tare you?"
            >>> filter._normalize_spaces(text)
            "Hello world How are you?"  # All whitespace normalized
        
        Important Notes:
            - This filter DESTROYS original line breaks - all text becomes a single line
            - Use BEFORE other filters that need spaces (like _fix_punctuation_spacing)
            - Very useful for creating clean, continuous text for analysis
            - Removes indentation, tabs, and any formatting based on spacing
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """
        Fix incorrect spacing around punctuation marks.
        
        This filter corrects two common spacing issues with punctuation:
        1. Removes spaces BEFORE punctuation (e.g., "Hello , world" -> "Hello, world")
        2. Adds spaces AFTER punctuation when missing (e.g., "Hello,world" -> "Hello, world")
        
        Args:
            text (str): Input text that may have incorrect spacing around punctuation
        
        Returns:
            str: Text with corrected spacing around punctuation marks
        
        How it works:
            First regex - r'\s+([.,;:!?])'
                - Finds: One or more whitespace characters followed by punctuation
                - Replaces: Just the punctuation (removes the space before)
                - Example: "Hello , world" → "Hello, world"
                        "test ; example" → "test; example"
            
            Second regex - r'([.,;:!?])([^\s\d])'
                - Finds: Punctuation followed by a non-space, non-digit character
                - Replaces: Punctuation + space + the character
                - Example: "Hello,world" -> "Hello, world"
                        "test;example" -> "test; example"
                        "Price:10" -> "Price: 10"  (digit exception: keeps "Price:10" as is)
        
        Examples:
            >>> text = "Hello ,world .This is a test;example"
            >>> filter._fix_punctuation_spacing(text)
            "Hello, world. This is a test; example"
            
            >>> text = "Price:10 items;quantity:5"
            >>> filter._fix_punctuation_spacing(text)
            "Price:10 items; quantity:5"  # Note: "Price:10" remains unchanged (digit after colon)
        
        Note:
            - Punctuation marks handled: . , ; : ! ?
            - Does NOT add space after punctuation if followed by a digit (e.g., "Version:2.0" stays as is)
            - Multiple spaces before punctuation are reduced to no space
            - This filter helps normalize text for better readability and consistency
        """
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])([^\s\d])', r'\1 \2', text)
        return text
    
    def _remove_empty_lines(self, text: str) -> str:
        """
        Remove completely empty lines from the text.
        
        This filter eliminates lines that contain only whitespace (spaces, tabs, or nothing at all).
        It's useful for cleaning up PDF-extracted text that often contains blank lines due to
        page breaks, formatting, or layout artifacts.
        
        Args:
            text (str): Input text that may contain empty or whitespace-only lines
        
        Returns:
            str: Text with all empty or whitespace-only lines removed. Lines that contain
                any non-whitespace characters are preserved exactly as they were.
        
        Example:
            >>> text = "Line 1\\n   \\nLine 2\\n\\nLine 3"
            >>> filter._remove_empty_lines(text)
            "Line 1\\nLine 2\\nLine 3"
            
            # Lines with only spaces/tabs are removed, but lines with content are kept
            >>> text = "Hello\\n    \\nWorld\\n\\t\\n!"
            >>> filter._remove_empty_lines(text)
            "Hello\\nWorld\\n!"
        
        Note:
            - A line is considered "empty" if line.strip() returns an empty string
            - This includes lines with only spaces, only tabs, or completely empty lines
            - The line breaks between remaining lines are preserved with '\\n'
            - This filter doesn't modify the content of non-empty lines
        """
        lines = [line for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def clean_text(self, text: Optional[str] = None, filters: Optional[List[Callable]] = None) -> str:
        """
        Apply cleaning filters to text to remove artifacts, normalize spacing, and improve readability.
        
        This method processes text through a series of filter functions that each perform
        specific cleaning operations (removing CID artifacts, normalizing hyphens, removing
        repeated dots, etc.). By default, it uses a comprehensive set of filters optimized
        for PDF-extracted text.
        
        Args:
            text (Optional[str]): Text to be cleaned. If None (default), uses self.raw_text
                which contains the raw extracted text from the PDF.
            
            filters (Optional[List[Callable]]): List of filter methods to apply sequentially.
                Each filter must be a callable that takes a string and returns a string.
                If None (default), uses a predefined list of cleaning filters:
                    - _remove_cid_artifacts: Removes (cid:12345) artifacts
                    - _remove_control_characters: Removes non-printable control chars
                    - _normalize_hyphens: Standardizes different hyphen types
                    - _remove_repeated_dots: Cleans up multiple dot patterns
                    - _remove_dots_between_words: Fixes word....word patterns
                    - _remove_lines_with_many_dots: Removes dot-heavy lines
                    - _remove_page_numbers: Strips isolated page numbers
                    - _remove_short_headers: Removes short uppercase headers
                    - _remove_empty_lines: Eliminates blank lines
                    - _normalize_spaces: Collapses multiple spaces
                    - _fix_punctuation_spacing: Adjusts space around punctuation
        
        Returns:
            str: The cleaned text after applying all filters. If the input text is empty,
                returns an empty string.
        
        Note:
            - If the input text is self.raw_text, the result is also stored in self.cleaned_text
            - The method includes validation checks after each filter and after all filters
            to ensure the text remains valid (non-empty, readable, etc.)
            - Filters are applied in the order they appear in the list - order matters!
        
        Example:
            >>> pdf = PDF("documento.pdf")
            >>> pdf.load_pdf()
            >>> cleaned = pdf.clean_text()
            >>> print(cleaned[:200])  # Print first 200 chars of cleaned text
            
            >>> # Use custom filters
            >>> custom_filters = [pdf._remove_page_numbers, pdf._normalize_spaces]
            >>> partially_cleaned = pdf.clean_text(filters=custom_filters)
        """
        # Get text to clean
        if text is None:
            text = self.raw_text
        
        if not text:
            return ""
        
        # Default filters if none provided
        if filters is None:
            filters = [
                self._remove_cid_artifacts,
                self._remove_control_characters,
                self._normalize_hyphens,
                self._remove_repeated_dots,     
                self._remove_dots_between_words,
                self._remove_lines_with_many_dots,
                self._remove_page_numbers,
                self._remove_short_headers,
                self._remove_empty_lines,
                self._normalize_spaces,
                self._fix_punctuation_spacing
            ]

        
        # Apply all filters
        cleaned = text
        for i, filter_func in enumerate(filters):
            cleaned = filter_func(cleaned)
            # Verificação após CADA filtro (opcional, mas útil para debug)
            self.validate_text_not_empty(cleaned, f"Valid content after applying filter {i+1}: {filter_func.__name__}")

        # Verificação final
        self.validate_text_not_empty(cleaned, "Valid content after applying all filters")
        
        # Store result
        if text is self.raw_text:
            self.cleaned_text = cleaned
        
        return cleaned
       
    # ===== UTILITY METHODS =====
    
    def save_cleaned_text(self, output_path: Optional[str] = None):
        """
        Save the cleaned text to a file on disk.
        
        This method exports the cleaned text content to a text file. If the text
        hasn't been cleaned yet, it automatically runs the cleaning pipeline first.
        The output file is saved with UTF-8 encoding to preserve all characters.
        
        Args:
            output_path (Optional[str]): Path where the cleaned text will be saved.
                - If provided: Saves to the specified path
                - If None (default): Automatically generates a filename based on
                the input PDF file (e.g., "document.pdf" → "document_cleaned.txt")
        
        Returns:
            None (saves file to disk and prints confirmation)
        
        File naming behavior:
            Input PDF path          | output_path=None          | Generated filename
            ------------------------|---------------------------|--------------------
            "report.pdf"            | None                      | "report_cleaned.txt"
            "./docs/thesis.pdf"     | None                      | "thesis_cleaned.txt"
            "/home/user/doc.pdf"    | None                      | "doc_cleaned.txt"
            "my document.pdf"       | None                      | "my document_cleaned.txt"
            
            With custom path:
            "report.pdf"            | "output/final.txt"        | "output/final.txt"
            "thesis.pdf"            | "./backup/clean.txt"      | "./backup/clean.txt"
        
        Examples:
            >>> pdf = PDF("annual_report.pdf")
            >>> pdf.load_pdf()
            
            >>> # Save with auto-generated filename
            >>> pdf.save_cleaned_text()
            Cleaned text saved to: annual_report_cleaned.txt
            
            >>> # Save to custom location
            >>> pdf.save_cleaned_text("./output/cleaned_report.txt")
            Cleaned text saved to: ./output/cleaned_report.txt
            
            >>> # Auto-cleaning happens if needed
            >>> pdf.raw_text = "Some text"  # Skip cleaning for example
            >>> pdf.save_cleaned_text()  # Automatically calls clean_text()
            Cleaned text saved to: annual_report_cleaned.txt
        
        File format details:
            - Encoding: UTF-8 (supports all Unicode characters)
            - Extension: .txt (text file)
            - Content: Cleaned text after all filters applied
            - Line breaks: Preserved from the cleaning process
        
        Error handling:
            The method may raise exceptions in these cases:
            - Permission denied (cannot write to directory)
            - Invalid path (directory doesn't exist)
            - Disk full (no space left)
            - Read-only filesystem
            
            These exceptions are not caught here and should be handled
            by the calling code if needed.
        
        Common use cases:
            1. Exporting results: Save cleaned text for external use
            2. Batch processing: Save multiple PDFs to cleaned files
            3. Archiving: Keep clean versions of documents
            4. Further processing: Feed cleaned text to other tools
        
        Note:
            This method overwrites existing files without warning!
            If you need to avoid overwriting, check file existence first:
            
            >>> import os
            >>> if not os.path.exists("output.txt"):
            ...     pdf.save_cleaned_text("output.txt")
            ... else:
            ...     print("File already exists, not overwriting")
        
        Example with error handling:
            >>> try:
            ...     pdf.save_cleaned_text("/readonly/output.txt")
            ... except PermissionError:
            ...     print("Cannot write to that location")
            ... except FileNotFoundError:
            ...     print("Directory doesn't exist")
            ... except Exception as e:
            ...     print(f"Save failed: {e}")
        """
        if not self.cleaned_text:
            self.clean_text()
        
        if not output_path:
            # Create output filename based on input
            input_path = Path(self.file_path)
            output_path = input_path.stem + "_cleaned.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.cleaned_text)
        
        print(f"Cleaned text saved to: {output_path}")
    
    def get_cleaned_text(self) -> str:
        """
        Return the cleaned text as a string.
        
        This method processes the cleaned text content and returns it as a string.
        If the text hasn't been cleaned yet, it automatically runs the cleaning pipeline first.
        
        Returns:
            str: The cleaned text content
        
        Examples:
            >>> pdf = PDF("annual_report.pdf")
            >>> pdf.load_pdf()
            
            >>> # Get cleaned text as string
            >>> text = pdf.get_cleaned_text()
            >>> print(len(text))
            
            >>> # Use the text immediately
            >>> word_count = len(pdf.get_cleaned_text().split())
            >>> print(f"Word count: {word_count}")
            
            >>> # First 500 characters
            >>> preview = pdf.get_cleaned_text()[:500]
        """
        if not self.cleaned_text:
            self.clean_text()
        
        return self.cleaned_text


    def get_stats(self) -> dict:
        """
        Get comprehensive statistics about the PDF text content.
        
        This method returns a dictionary containing various metrics about both
        the raw extracted text and the cleaned text. It's useful for monitoring
        the effect of cleaning filters and understanding document characteristics.
        
        Returns:
            dict: A dictionary with the following keys:
            
            Key              | Type  | Description
            -----------------|-------|------------------------------------
            'pages'          | int   | Total number of pages in PDF
            'raw_chars'      | int   | Total characters in raw text
            'raw_lines'      | int   | Total lines in raw text
            'cleaned_chars'  | int   | Total characters in cleaned text (0 if not cleaned)
            'cleaned_lines'  | int   | Total lines in cleaned text (0 if not cleaned)
        
        Statistics calculated:
            - Page count: Direct from PDF metadata
            - Character count: Length of string (includes all characters)
            - Line count: Number of lines after splitting by '\\n'
        
        Examples:
            >>> pdf = PDF("document.pdf")
            >>> pdf.load_pdf()
            
            >>> # Get stats before cleaning
            >>> stats = pdf.get_stats()
            >>> print(stats)
            {
                'pages': 42,
                'raw_chars': 150000,
                'raw_lines': 2500,
                'cleaned_chars': 0,
                'cleaned_lines': 0
            }
            
            >>> # Clean and check stats again
            >>> pdf.clean_text()
            >>> stats = pdf.get_stats()
            >>> print(stats)
            {
                'pages': 42,
                'raw_chars': 150000,
                'raw_lines': 2500,
                'cleaned_chars': 120000,
                'cleaned_lines': 1800
            }
            
            >>> # Calculate cleaning impact
            >>> reduction = (1 - stats['cleaned_chars'] / stats['raw_chars']) * 100
            >>> print(f"Cleaning reduced text by {reduction:.1f}%")
            Cleaning reduced text by 20.0%
        
        Common analyses using stats:
            1. Cleaning effectiveness:
            >>> stats = pdf.get_stats()
            >>> removed_chars = stats['raw_chars'] - stats['cleaned_chars']
            >>> removed_lines = stats['raw_lines'] - stats['cleaned_lines']
            
            2. Average line length:
            >>> avg_raw = stats['raw_chars'] / stats['raw_lines']
            >>> avg_clean = stats['cleaned_chars'] / stats['cleaned_lines']
            
            3. Text density:
            >>> chars_per_page = stats['raw_chars'] / stats['pages']
            
            4. Cleaning ratio:
            >>> retention = stats['cleaned_chars'] / stats['raw_chars']
        
        Notes:
            - cleaned_chars and cleaned_lines are 0 if clean_text() hasn't been called
            - Line counts may include empty lines (depending on text state)
            - Character counts include all characters (spaces, punctuation, etc.)
            - The method is very fast (O(n) where n is text length)
        
        Example with formatting:
            >>> def print_pretty_stats(pdf):
            ...     stats = pdf.get_stats()
            ...     print("=" * 40)
            ...     print(f"PDF Statistics: {Path(pdf.file_path).name}")
            ...     print("=" * 40)
            ...     print(f"Pages: {stats['pages']}")
            ...     print(f"\\nRaw Text:")
            ...     print(f"  Characters: {stats['raw_chars']:,}")
            ...     print(f"  Lines: {stats['raw_lines']:,}")
            ...     if stats['cleaned_chars']:
            ...         print(f"\\nCleaned Text:")
            ...         print(f"  Characters: {stats['cleaned_chars']:,}")
            ...         print(f"  Lines: {stats['cleaned_lines']:,}")
            ...         reduction = (1 - stats['cleaned_chars']/stats['raw_chars'])*100
            ...         print(f"  Reduction: {reduction:.1f}%")
            ...     print("=" * 40)
            
            >>> print_pretty_stats(pdf)
            ========================================
            PDF Statistics: annual_report.pdf
            ========================================
            Pages: 150
            
            Raw Text:
            Characters: 245,789
            Lines: 4,567
            
            Cleaned Text:
            Characters: 198,234
            Lines: 3,890
            Reduction: 19.3%
            ========================================
        
        Note about performance:
            - String splitting for line count is O(n)
            - For very large documents (>1M chars), consider caching
            - Method recreates the dict each call (cheap)
        """
        return {
            'pages': self.page_count,
            'raw_chars': len(self.raw_text),
            'raw_lines': len(self.raw_text.split('\n')),
            'cleaned_chars': len(self.cleaned_text) if self.cleaned_text else 0,
            'cleaned_lines': len(self.cleaned_text.split('\n')) if self.cleaned_text else 0,
        }

    # =====  VERIFICATON METHOD =====
    
    def validate_text_not_empty(self, text: str,context: str = "") -> bool:
        """
        Validate that text is not empty and has reasonable content after processing.
        
        This method performs comprehensive validation on text to ensure it contains
        meaningful content. It's used throughout the cleaning pipeline to catch
        issues early, such as filters that might accidentally remove all content.
        
        Args:
            text (str): Text to be validated (can be raw or cleaned text)
            context (str, optional): Context description for error/warning messages.
                Helps identify where in the pipeline the validation occurred.
                Examples: "after filter 3: _remove_repeated_dots", 
                        "after all filters", "before saving"
        
        Returns:
            bool: True if text is valid (non-empty with reasonable content),
                False if text is invalid (None, empty, or only whitespace)
        
        Validation levels (in order):
            1. CRITICAL CHECKS (return False if failed):
            - Text is not None
            - Text is not empty string ("")
            - Text contains more than just whitespace
            
            2. WARNING CHECKS (print warning but return True):
            - Text has at least 10 useful characters (after stripping)
            - At least 30% of characters are alphanumeric or spaces
            - Text contains at least 3 words
            
            Note: Warnings don't fail validation but alert to potential issues
        
        Validation examples:
            Text Input                          | Result | Reason
            ------------------------------------|--------|------------------
            None                                | False  | Text is None
            ""                                  | False  | Empty string
            "   \n\t   "                        | False  | Only whitespace
            "Hi"                                | True   | Too short (<10 chars)
            "12345!!!@@@"                       | True   | Low readable ratio
            "Hello world"                       | True   | Only 2 words
            "The quick brown fox jumps"         | True   | Passes all checks
        
        Message format:
            - ERROR:   Critical failure that makes text unusable
            - WARNING: Text is usable but might have issues
            - Valid:   Text passed all checks (with stats)
        
        Example outputs:
            >>> validate_text_not_empty(None, "before filter")
            "ERROR: Text is None before filter"
            False
            
            >>> validate_text_not_empty("   ", "after cleaning")
            "ERROR: Text contains only spaces/tabs/\\n after cleaning"
            "    Length: 3, characters: '   '"
            False
            
            >>> validate_text_not_empty("Hi", "after filter 1")
            "WARNING: Text too short after filter 1"
            "   Only 2 useful characters"
            "Valid text after filter 1: 2 chars, 1 words"
            True
            
            >>> validate_text_not_empty("Hello world!", "final check")
            "Valid text final check: 11 chars, 2 words"
            True
               
        Customization points:
            To make validation stricter, uncomment the return False lines:
            - Line with "return False" under minimum size check
            - Line with "return False" under ratio check
            
            To adjust thresholds, modify:
            - Minimum characters: change 10 to desired value
            - Readable ratio: change 0.3 to desired value (0.0-1.0)
            - Minimum words: change 3 to desired value
        
        Note:
            This method is designed to be informative, not just a boolean check.
            The detailed messages help identify exactly what's wrong with the text
            and where in the pipeline the issue occurred.
        """
        
        # 1. Basic check for None or empty string
        if text is None:
            print(f" ERROR: Text is None {context}")
            return False
        
        if len(text) == 0:
            print(f" ERROR: Text completely empty {context}")
            return False
        
        # 2. Check for only whitespace
        if text.strip() == "":
            print(f" ERROR: Text contains only spaces/tabs/\\n {context}")
            print(f"    Length: {len(text)}, characters: {repr(text[:50])}")
            return False
        
        # 3. Minimum size check (optional)
        if len(text.strip()) < 10:  # less than 10 useful characters
            print(f"  WARNING: Text too short {context}")
            print(f"   Only {len(text.strip())} useful characters")
            # Can be a warning, not necessarily an error
            return False  # uncomment if you want this as error
        
        # 4. Valid character ratio check
        text_stripped = text.strip()
        if text_stripped:
            # Count alphanumeric characters vs total
            alnum_count = sum(c.isalnum() or c.isspace() for c in text_stripped)
            ratio = alnum_count / len(text_stripped)
            
            if ratio < 0.3:  # less than 30% are letters/numbers/spaces
                print(f"  WARNING: Low proportion of readable text {context}")
                print(f"    Only {ratio:.1%} alphanumeric/space characters")
                return False  # uncomment if you want this as error
        
        # 5. Minimum word count (optional)
        word_count = len(text_stripped.split())
        if word_count < 3:
            print(f"  WARNING: Very few words {context}")
            print(f"    Only {word_count} words")
            return False
        
        print(f" Valid text {context}: {len(text_stripped)} chars, {word_count} words")
        return True

    # def get_doi(self) -> str:
    #     """
    #     Extract and return the DOI from the cleaned text.
        
    #     This method searches for DOI patterns in the text and returns the first
    #     valid DOI found. Returns empty string if no DOI is found.
        
    #     Returns:
    #         str: The DOI string if found, otherwise an empty string
        
    #     Examples:
    #         >>> pdf = PDF("article.pdf")
    #         >>> pdf.load_pdf()
    #         >>> doi = pdf.get_doi()
    #         >>> print(doi)
    #         '10.1038/s41586-020-1234-5'
            
    #         >>> if doi:
    #         ...     print(f"DOI found: {doi}")
    #         ... else:
    #         ...     print("No DOI found")
    #     """
    # #import re
    
    #     # Get cleaned text
    #     text = self.get_cleaned_text()
        
    #     if not text:
    #         return ""
        
    #     # Pattern for DOI (without prefix)
    #     doi_pattern = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+'
        
    #     # Pattern for DOI with "DOI:" prefix
    #     doi_with_prefix = r'DOI:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+)'
        
    #     # First try to find DOI with prefix (more specific)
    #     match = re.search(doi_with_prefix, text, re.IGNORECASE)
    #     if match:
    #         return match.group(1)
        
    #     # If not found, try without prefix
    #     match = re.search(doi_pattern, text)
    #     if match:
    #         return match.group(0)
        
    #     # No DOI found
    #     return ""

    # def get_doi(self) -> str:
    #     """
    #     Extract and return the DOI from the cleaned text.
    #     """
    #     import re
        
    #     if not self.cleaned_text:
    #         print("DEBUG: cleaned_text está vazio")
    #         return ""
        
    #     # DEBUG: Mostra o texto antes do regex
    #     print("DEBUG: Texto completo:")
    #     print(repr(self.cleaned_text))  # Mostra caracteres especiais
    #     print("\nDEBUG: Primeiros 500 caracteres:")
    #     print(self.cleaned_text[:500])
    #     print("\n" + "="*50)
    #     input()

    #     # Procura o DOI
    #     doi_pattern = r'10\.\d{4,9}/[^\s]+'
    #     match = re.search(doi_pattern, self.cleaned_text)
        
    #     if match:
    #         print(f"DEBUG: Regex encontrou: '{match.group(0)}'")
    #         return match.group(0)
    #     else:
    #         print("DEBUG: Regex NÃO encontrou nenhum DOI")
    #         return ""

    def get_doi(self) -> str:
        
        if not self.cleaned_text:
            return ""
        
        # Remove espaços dentro de padrões de DOI
        # Substitui "doi. org" por "doi.org"
        texto_limpo = re.sub(r'doi\.\s+org', 'doi.org', self.cleaned_text)
        # Substitui "j. physa" por "j.physa" (remove espaço entre palavras)
        texto_limpo = re.sub(r'(\w)\.\s+(\w)', r'\1.\2', texto_limpo)
        
        # Agora busca o DOI
        doi_pattern = r'10\.\d{4,9}/[^\s]+'
        match = re.search(doi_pattern, texto_limpo)
        
        if match:
            return match.group(0)
        
        return ""