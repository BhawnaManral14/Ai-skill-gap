"""
Resume Parser Module for AI Skill-Gap & Job Readiness Predictor.
Extracts and normalizes text from PDF and TXT resume files with robust error handling.
"""

import io
import re
from typing import Optional
from pypdf import PdfReader
from pypdf.errors import PdfStreamError, EmptyFileError


class ResumeParsingError(Exception):
    """Custom exception raised when resume text extraction fails."""
    pass


def normalize_text(text: str) -> str:
    """
    Cleans and standardizes extracted resume text.
    Removes excessive whitespace, standardizes line breaks, and strips non-printable characters.
    """
    if not text:
        return ""
    
    # Replace non-breaking spaces and other special space characters
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Replace 3 or more consecutive newlines with 2 newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    
    # Strip leading and trailing whitespace
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from multi-page PDF documents using pypdf.
    
    Args:
        file_bytes: Raw binary bytes of the PDF file.
        
    Returns:
        Normalized extracted string content.
        
    Raises:
        ResumeParsingError: If PDF is corrupt, password-protected, or unreadable.
    """
    if not file_bytes:
        raise ResumeParsingError("The provided PDF file is empty (0 bytes).")

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        # Check for encryption
        if reader.is_encrypted:
            try:
                # Attempt empty password decryption
                reader.decrypt("")
            except Exception:
                raise ResumeParsingError(
                    "The PDF file is encrypted and password-protected. Please provide an unencrypted PDF."
                )

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ResumeParsingError("The PDF document does not contain any pages.")

        extracted_pages = []
        for page_idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            except Exception as page_err:
                # Log or tolerate single page extraction issues if other pages work
                continue

        full_text = "\n\n".join(extracted_pages)
        normalized = normalize_text(full_text)

        if not normalized or len(normalized) < 20:
            raise ResumeParsingError(
                "Could not extract sufficient text from this PDF. "
                "The file may be a scanned image without an OCR text layer."
            )

        return normalized

    except (PdfStreamError, EmptyFileError) as e:
        raise ResumeParsingError(f"Corrupt or invalid PDF file stream: {str(e)}")
    except ResumeParsingError:
        raise
    except Exception as e:
        raise ResumeParsingError(f"Failed to process PDF file: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extracts and decodes text from raw bytes of a TXT file using UTF-8 with fallback encodings.
    
    Args:
        file_bytes: Raw binary bytes of the TXT file.
        
    Returns:
        Normalized string content.
        
    Raises:
        ResumeParsingError: If text cannot be decoded or is empty.
    """
    if not file_bytes:
        raise ResumeParsingError("The provided text file is empty (0 bytes).")

    # Try common encodings in order
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    decoded_text: Optional[str] = None

    for enc in encodings:
        try:
            decoded_text = file_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if decoded_text is None:
        # Final fallback with replacement characters
        try:
            decoded_text = file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise ResumeParsingError(f"Failed to decode text file: {str(e)}")

    normalized = normalize_text(decoded_text)
    if not normalized or len(normalized) < 10:
        raise ResumeParsingError("The text file is empty or contains no readable text content.")

    return normalized


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Primary interface for resume text extraction.
    Automatically detects file format by filename extension (.pdf or .txt) and extracts content.
    
    Args:
        file_bytes: Raw binary bytes of the resume file.
        filename: Name of the uploaded file including extension.
        
    Returns:
        Clean, normalized plain text string of the resume.
        
    Raises:
        ResumeParsingError: If file format is unsupported or extraction fails.
    """
    if not filename:
        raise ResumeParsingError("Filename must be provided.")

    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ResumeParsingError(
            f"Unsupported file format for '{filename}'. Supported formats are: .pdf, .txt"
        )
