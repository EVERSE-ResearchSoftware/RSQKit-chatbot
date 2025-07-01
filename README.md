# RSQKit Chatbot
A Retrieval-Augmented Generation (RAG) chatbot connected to the [RSQKit](https://everse.software/RSQKit/). 
Built with Streamlit for the web interface, ChromaDB for vector storage, Ollama for local model serving, and supports multiple AI providers via OpenAI-compatible APIs.

## 📌 Features  
- **Retrieval‑Augmented Generation (RAG) Chatbot**  
  Leverages document retrieval and generative AI to provide accurate, context‑aware answers.

- **Multi‑Provider AI Support**  
  Compatible with Ollama and any OpenAI‑style API for both language models and embedding services. 
  Set default models, and API credentials without modifying the code using the `provider_config.yaml` file.

- **Hybrid Search Engine**  
  Integrates BM25 keyword search with vector‑based similarity to ensure precise and relevant results.

- **Streamlit‑Powered Interface**  
  Interactive web app featuring chat history, source traceability, and real‑time sidebar controls.

- **Modular Architecture**  
  Clean separation of ingestion, configuration, and application logic for easy extension and maintenance.


# 📦 Installation

## Prerequisites

Before you begin, please make sure you’ve installed all of the following. Ollama is only needed if you plan to run models locally. If that’s the case and you haven’t installed it yet, head over to the official docs and follow the platform-specific instructions: https://ollama.com/



- **Python 3.10 or higher**

- **Ollama** (used for running models locally)  
  To install Ollama, follow the official instructions here: [https://ollama.com](https://ollama.com)  
  After installation, confirm it's working by running:
  ```bash
    ollama --help
  ```

* **API Key** from an AI provider that supports the OpenAI API protocol (required for using remote models)

* **Embedding Model** for semantic search (e.g., `bge-m3` with Ollama).
  Install it locally with:

  ```bash
  ollama pull bge-m3
  ```

* **LLM** for text generation (e.g., `deepseek-r1:14b` with Ollama).
  Install it locally with:

  ```bash
  ollama pull deepseek-r1:14b
  ```

* **Reranker** *(optional)* — improves ranking accuracy of search results.

---

## Configuring AI Providers

To integrate an AI provider, update the `provider_config.yaml` file with the provider's details. Below is an example configuration format:

  ```yaml
  new_provider:
    name: "Example Provider"
    base_url: "https://api.exampleprovider.com/v1"
    base_url_vision: "https://api.exampleprovider.com/v1/chat/completions"
    rerank_url: "https://api.exampleprovider.com/v1/rerank"
    api_env_var: "API_KEY_EXAMPLE_PROVIDER"
    fall_back_provider: "ollama"
    supports_embedding: true
    supports_reranker: true
    models:
      default_embedding: "example-embedding-model"
      default_reranker: "example-reranker-model"
      default_llm: "example-llm-model"
      default_vision: "example-vision-model"
  ```


## Steps

1. **Clone the repository**  
   Open your terminal and run:
    ```bash
    git clone https://github.com/EVERSE-ResearchSoftware/RSQKit-chatbot.git
    cd RSQKit-chatbot
    ```

2. **Set up a virtual environment**

   Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   With the virtual environment activated, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **⚠️ Configure environment variables**

   Create a `.env` file and populate it with your API keys and other required settings as referenced in `provider_config.yaml`.


# 🚀 Usage

## 1. Make these Bash Files Executable
   ```sh
   chmod +x download_rsqkit_files.sh ingest_rsqkit_files.sh run_app.sh
   ```
## 2. Download and Ingest RSQKit Files

  ```sh
  ./download_rsqkit_files.sh  # This will take a moment. Skip if you already have the files in the directory `rsqkit_markdown`.
  ./ingest_rsqkit_files.sh    # This will also take a moment. Ensure you have an embedding model available.
  ```
This will download RSQKit files and ingest them into ChromaDB for retrieval.
## 3. Run the Chatbot App
  ```sh
  ./run_app.sh
  ```

Opens a Streamlit web interface at `http://localhost:8501`. 
# 📁 Project Structure
```
rsqkit-chatbot/
├── app.py                              # Main Streamlit app
├── chroma_data_ingestor.py
├── core_utils                          # Containing utility functions (data processing, retrieval)
├── directories.yaml
├── download_rsqkit_files.sh
├── ingest_rsqkit_files.sh
├── llm_provider_tools.py
├── llms                                # Contains LLM chat functions for OpenAI protocol
├── pages                               # Contains the other pages of the app - Streamlit protocol
├── prompt_templates
├── provider_config.yaml                # ⚠️ Where you setup AI provider, API_KEY_VAR, LLM, embedding, etc
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements-test.txt
├── requirements.txt
├── rsqkit_scrap.py
├── rsqkit_markdown                     # Contains markdown input files (not tracked)
├── run_app.sh
├── settings.py
├── static
├── task_modules
├── templates
├── tests
├── ui
├── uv.lock
└── vision                              # Contains function for Vision Language Models
```
# 📝 Contributing

We welcome contributions of all kinds!

- **Bug Reports**: If you encounter an issue, please open a GitHub issue with a clear and detailed description.
- **Feature Requests**: Have an idea for an enhancement? Open a discussion or submit a pull request with your proposal.
- **Code Contributions**: Fork the repository, make your changes, and submit a pull request. Make sure to follow any existing coding conventions and include relevant documentation or tests where applicable.

# 📄 License

This project is licensed under the Apache-2.0 License.  
See the [LICENSE](./LICENSE) file for full details.

# 🙏 Acknowledgments

- This project is financially supported by [EVERSE](https://everse.software/), [IJCLAB](https://www.ijclab.in2p3.fr), [IN2P3](https://www.in2p3.cnrs.fr), and [CNRS](https://www.cnrs.fr)
- We extend our gratitude to the teams at [AI4EOSC](https://docs.ai4eosc.eu/en/latest/reference/llm.html), [Albert API](https://github.com/etalab-ia/albert-api) and [RAGaRenn](https://ragarenn.eskemm-numerique.fr/index.html) for generously providing their AI resources during the development of this application.
- The project draws inspiration from Retrieval-Augmented Generation (RAG) and hybrid search strategies in modern AI engineering.
- The application is built using tools like [Streamlit](https://streamlit.io), [ChromaDB](https://www.trychroma.com), and [Ollama](https://ollama.com).
