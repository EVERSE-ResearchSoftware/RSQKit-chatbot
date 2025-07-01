import streamlit as st
from itertools import zip_longest
from pathlib import Path
import base64
import shutil


def create_browser_viewable_link(file_path, display_text, size_limit_bytes=2_000_000):
    """
    Affiche le fichier dans un iframe si petit, sinon copie dans static/documents/ et l'affiche par URL.
    Compatible avec PDF, images, HTML, etc.

    Args:
        file_path (str): Chemin vers le fichier local (ex: 'documents/monfichier.pdf')
        display_text (str): Texte du lien ou titre du fichier
        size_limit_bytes (int): Taille max pour base64 embedding (défaut: 2 Mo)

    Returns:
        str: HTML contenant l'iframe ou un lien de téléchargement
    """
    path = Path(file_path)
    if not path.exists():
        return f'<span style="color: red;">{display_text} (fichier introuvable)</span>'

    file_extension = path.suffix.lower()
    browser_viewable = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".txt": "text/plain",
        ".html": "text/html",
        ".htm": "text/html",
    }

    try:
        with open(file_path, "rb") as file:
            file_bytes = file.read()

        # ✅ CASE 1: Embed as base64 if small enough and viewable
        if file_extension in browser_viewable and len(file_bytes) <= size_limit_bytes:
            mime_type = browser_viewable[file_extension]
            b64 = base64.b64encode(file_bytes).decode()
            return f"""
            <div style="margin-top: 1em;">
                <div style="font-weight: bold; margin-bottom: 0.5em;">🔍 {display_text}</div>
                <iframe src="data:{mime_type};base64,{b64}" 
                        width="100%" height="600px" 
                        style="border: 1px solid #ccc; border-radius: 8px;">
                </iframe>
            </div>
            """

        # ✅ CASE 2: File too big → copy to static/documents and use URL
        static_target = Path("static/documents") / path.name
        static_target.parent.mkdir(parents=True, exist_ok=True)
        if not static_target.exists():
            shutil.copy(file_path, static_target)

        file_url = f"/static/documents/{path.name}"
        return f"""
        <div style="margin-top: 1em;">
            <div style="font-weight: bold; margin-bottom: 0.5em;">🔍 {display_text}</div>
            <iframe src="{file_url}" 
                    width="100%" height="600px" 
                    style="border: 1px solid #ccc; border-radius: 8px;">
            </iframe>
        </div>
        """

    except Exception as e:
        return f'<span style="color: red;">{display_text} (erreur: {str(e)})</span>'


def create_dual_link(file_path, display_text, page_start):
    """
    Crée deux liens : un pour ouvrir en ligne et un pour télécharger
    """
    http_file = f"http://localhost:8501/app/static/documents/{Path(file_path).name}#page={page_start}"
    if not file_path or not Path(file_path).exists():
        return f'<span style="color: red;">{display_text} (fichier introuvable)</span>'

    try:
        with open(file_path, "rb") as file:
            file_bytes = file.read()

        file_extension = Path(file_path).suffix.lower()
        filename = Path(file_path).name
        b64 = base64.b64encode(file_bytes).decode()

        # Types visualisables dans le navigateur
        browser_viewable = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".txt",
            ".html",
            ".svg",
        }

        if file_extension in browser_viewable:
            mime_types = {
                ".pdf": "application/pdf",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".txt": "text/plain",
                ".html": "text/html",
                ".svg": "image/svg+xml",
            }
            mime_type = mime_types.get(file_extension, "application/octet-stream")

            return f"""
            <div style="display: flex; gap: 10px; align-items: center;">
                <a href="{http_file}" 
                   target="_blank" 
                   style="text-decoration: none; color: #1f77b4;">
                   🔗 Ouvrir {display_text}
                </a>
                <span style="color: #ccc;">|</span>
                <a href="data:application/octet-stream;base64,{b64}" 
                   download="{filename}" 
                   style="text-decoration: none; color: #28a745;">
                   📄 Télécharger
                </a>
            </div>
            """
        else:
            # Seulement téléchargement pour les autres types
            return f"""<a href="data:application/octet-stream;base64,{b64}" 
                       download="{filename}" 
                       style="text-decoration: none; color: #ff6b6b;">
                       📄 {display_text} (télécharger uniquement)
                   </a>"""

    except Exception as e:
        return f'<span style="color: red;">{display_text} (erreur)</span>'


def view_sources(relevant_docs, metadatas, link_style="dual"):
    """
    Affiche les sources avec différents styles de liens

    Args:
        relevant_docs: Liste des documents (textes, extraits, etc.)
        metadatas: Liste des métadonnées associées
        link_style (str): "view", "download", ou "dual"
    """
    for idx, (doc, meta) in enumerate(
        zip_longest(relevant_docs, metadatas, fillvalue={}), start=1
    ):
        # Titre et contenu principal
        st.markdown(
            f'<div class="source-title">Source #{idx}</div>', unsafe_allow_html=True
        )
        st.success(doc)

        if meta:
            selected_keys = ["title"]
            file_path = meta.get("file-path")
            page_start = int(meta.get("page-start"))

            for key in selected_keys:
                value = meta.get(key)
                displayed_key = "Fichier" if key == "title" else key

                # Affiche le label (titre)
                st.markdown(
                    f'<div class="source-title">{displayed_key}: {value}</div>',
                    unsafe_allow_html=True,
                )

                # Affiche le lien selon le style choisi
                if file_path:
                    if link_style == "view":
                        file_link = create_browser_viewable_link(file_path, value)
                    elif link_style == "dual":
                        file_link = create_dual_link(file_path, value, page_start)
                    else:  # "download"
                        file_link = create_downloadable_link(file_path, value)

                    st.markdown(file_link, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<span style="color: gray;">Aucun fichier associé</span>',
                        unsafe_allow_html=True,
                    )

        st.markdown('<hr class="source-divider">', unsafe_allow_html=True)


def create_downloadable_link(file_path, display_text):
    """Fonction de téléchargement simple (de l'exemple précédent)"""
    if not file_path or not Path(file_path).exists():
        return f'<span style="color: red;">{display_text} (fichier introuvable)</span>'

    try:
        with open(file_path, "rb") as file:
            file_bytes = file.read()

        b64 = base64.b64encode(file_bytes).decode()
        filename = Path(file_path).name

        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="color: #1f77b4;">📄 {display_text}</a>'

    except Exception as e:
        return f'<span style="color: red;">{display_text} (erreur)</span>'


# Exemple d'utilisation
if __name__ == "__main__":
    # Données de test
    relevant_docs = ["Contenu du document 1", "Contenu du document 2"]
    metadatas = [
        {"title": "Document1.pdf", "file-path": "/path/to/doc1.pdf"},
        {"title": "Image.jpg", "file-path": "/path/to/image.jpg"},
    ]

    st.header("Style: Ouverture en onglet (quand possible)")
    view_sources(relevant_docs, metadatas, link_style="view")

    st.header("Style: Double lien (ouvrir + télécharger)")
    view_sources(relevant_docs, metadatas, link_style="dual")
