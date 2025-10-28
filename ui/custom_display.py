import streamlit as st
from itertools import zip_longest
from pathlib import Path
import base64
import html
import os
import shutil


def copy_to_static_folder(file_path):
    """
    Copy file to static/documents folder and return the HTTP URL
    This allows files to be opened in new tabs without browser blocking
    """
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            return None

        # Create static folder if it doesn't exist
        static_dir = Path("static/documents")
        static_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to static folder (only if not already there)
        target_path = static_dir / source_path.name
        if not target_path.exists():
            shutil.copy2(file_path, target_path)

        # Return the HTTP URL (Streamlit serves /app/static/ by default)
        return f"/app/static/documents/{source_path.name}"
    except Exception as e:
        print(f"Error copying to static folder: {e}")
        return None


def encode_file_to_base64(file_path):
    """Encode file to base64 once and cache it (for download button only)"""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        return base64.b64encode(file_data).decode()
    except Exception as e:
        return None


def create_dual_link_html(b64_data, filename, page_number, file_ext, http_url=None):
    """
    Creates HTML for view/download links
    Uses HTTP URL for viewing (opens in new tab) and base64 for download
    """
    if not b64_data:
        return '<span style="color: red;">Erreur de lecture du fichier</span>'

    # Web viewer link (only for PDFs) - uses HTTP URL to avoid browser blocking
    if file_ext.lower() == ".pdf" and http_url:
        viewer_url = f"{http_url}#page={page_number}"
        viewer_link = f"""<a href="{viewer_url}" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #2196F3; color: white; text-decoration: none; border-radius: 4px; margin: 5px;">🔎 View (page {page_number})</a>"""
    else:
        viewer_link = ""

    # Download link (uses base64 to force download)
    download_link = f"""<a href="data:application/octet-stream;base64,{b64_data}" download="{filename}" style="display: inline-block; padding: 8px 16px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin: 5px;">📥 Download</a>"""

    return viewer_link + download_link


def generate_sources_html(relevant_docs, metadatas, link_style="dual"):
    """
    Pre-computes source HTML with embedded base64 files
    Returns HTML string for immediate display
    """
    html_parts = []

    for idx, (doc, meta) in enumerate(
        zip_longest(relevant_docs, metadatas, fillvalue={}), start=1
    ):
        # Source title
        html_parts.append(
            f'<div class="source-title" style="font-weight: bold; margin: 15px 0 10px 0; font-size: 1.1em;">📄 Source #{idx}</div>'
        )

        # Document content (truncate and escape)
        doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
        escaped_doc = html.escape(doc_preview).replace("\n", "<br>")
        html_parts.append(
            f'<div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0;">{escaped_doc}</div>'
        )

        if meta:
            file_path = meta.get("file-path")
            page_start = int(meta.get("page-start", 1))
            title = meta.get("title", "Document")
            page_end = meta.get("page-end", "last_page")  # Addition of page_end

            # Display file title
            escaped_title = html.escape(title)
            html_parts.append(
                f'<div style="margin: 10px 0;">File: {escaped_title}</div>'
            )
            html_parts.append(
                f'<div style="margin: 0 0 10px 0;">Pages: {page_start} to {page_end}</div>'
            )
            # Generate file links
            if file_path and Path(file_path).exists():
                # Copy file to static folder for HTTP serving
                http_url = copy_to_static_folder(file_path)

                # Encode file for download button
                b64_data = encode_file_to_base64(file_path)
                file_ext = Path(file_path).suffix
                filename = Path(file_path).name

                if b64_data:
                    file_links = create_dual_link_html(
                        b64_data, filename, page_start, file_ext, http_url
                    )
                    html_parts.append(
                        f'<div style="margin: 10px 0;">{file_links}</div>'
                    )
                else:
                    html_parts.append(
                        '<span style="color: red;">Erreur lors de la lecture du fichier</span>'
                    )
            else:
                html_parts.append(
                    '<span style="color: gray;">Aucun fichier associé</span>'
                )

        # Divider
        html_parts.append(
            '<hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">'
        )

    return "\n".join(html_parts)


def view_sources(relevant_docs, metadatas, link_style="dual"):
    """
    Legacy function for backward compatibility
    Displays sources directly (not used for chat history)
    """
    sources_html = generate_sources_html(relevant_docs, metadatas, link_style)
    st.markdown(sources_html, unsafe_allow_html=True)
