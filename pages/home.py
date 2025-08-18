import streamlit as st
from app_config import display_navigation_links
from app_config import ICONS


def main():
    display_navigation_links()
    # Custom CSS for beautiful card design
    st.markdown(
        """
    <style>
        .main-title {
            text-align: center;
            color: #2E4053;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            text-align: center;
            color: #5D6D7E;
            font-size: 1.2rem;
            margin-bottom: 3rem;
            font-weight: 300;
        }
        
        /* Style for Streamlit buttons to look like cards */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: none;
            color: white;
            min-height: 200px;
            width: 100%;
            font-size: 1rem;
            font-weight: 600;
        }
        
        .stButton > button:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        
        .stButton > button:focus {
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            transform: translateY(-10px);
        }
        
        /* Different gradients for each card */
        .card-general .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .card-rag .stButton > button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .card-document .stButton > button {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .card-collections .stButton > button {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }
        
        .card-rag-collections .stButton > button {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        
        .service-icon {
            font-size: 4rem;
            display: block;
            margin-bottom: 1rem;
        }
        
        .service-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .service-description {
            font-size: 1rem;
            opacity: 0.95;
            line-height: 1.4;
            display: block;
        }
        
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
            
            .stButton > button {
                padding: 1.5rem;
                min-height: 180px;
            }
            
            .service-icon {
                font-size: 3rem;
            }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main title and subtitle
    st.markdown(
        '<h1 class="main-title">Generative AI Services</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Choose a service to get started</p>',
        unsafe_allow_html=True,
    )

    # Create columns for the service cards
    col1, col2 = st.columns(2)
    col3, col4, col5 = st.columns(3)

    # Service cards using Streamlit buttons with custom styling
    with col1:
        st.markdown('<div class="card-general">', unsafe_allow_html=True)
        if st.button(
            f'{ICONS["chat"]}\n\n**Chat**\n\nVersatile AI assistant for general inquiries.',
            key="general_card",
            use_container_width=True,
        ):
            st.switch_page("pages/general_purpose_chat.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-rag">', unsafe_allow_html=True)
        if st.button(
            f'{ICONS["rsqkit_chat"]}\n\n**RSQKit Chat**\n\n',
            key="rag_card",
            use_container_width=True,
        ):
            st.switch_page("pages/rsqkit_rag_chat.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-document">', unsafe_allow_html=True)
        if st.button(
            f'{ICONS["document_chat"]}\n\n**Document Chat**\n\nAsk questions on your temporary document(s)',
            key="document_card",
            use_container_width=True,
        ):
            st.switch_page("pages/document_chat.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card-collections">', unsafe_allow_html=True)
        if st.button(
            f'{ICONS["collections"]}\n\n**Collections**\n\nManage and organize your document collections',
            key="collections_card",
            use_container_width=True,
        ):
            st.switch_page("pages/collections.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="card-rag-collections">', unsafe_allow_html=True)
        if st.button(
            f'{ICONS["rag_collections"]}\n\n**RAG on Collection**\n\nAsk questions on multiple documents',
            key="rag_collections_card",
            use_container_width=True,
        ):
            st.switch_page("pages/rag_on_collection.py")
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #7F8C8D; font-size: 0.9rem; margin-top: 2rem;">Build by Hugo Bacard. With L.O.V.E </p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()