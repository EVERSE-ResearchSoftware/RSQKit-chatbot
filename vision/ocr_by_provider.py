import base64
import requests
import json
from PIL import Image
from dotenv import load_dotenv
from typing import Any

load_dotenv()


# from pdf2image import convert_from_path

# def pdf_to_images(pdf_path, dpi=200):
#     """
#     Convert a multi-page PDF into a list of PIL images (one per page).
#     Returns the list of images.
#     """
#     return convert_from_path(pdf_path, dpi=dpi)


def image_to_base64(pil_image):
    """
    Convert a PIL image to a base64-encoded PNG bytes string.
    """
    import io

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str


def process_response(response):
    """
    Traite la réponse et imprime le contenu extrait.

    Args:
    response: L'objet de réponse contenant le contenu à traiter.
    """
    response_string = response.content.decode()
    lines_response = response_string.strip().split("\n")

    # Iterate over each line
    for line in lines_response:
        # Skip the [DONE] line
        if line == "data: [DONE]":
            break
        # Remove the "data: " prefix
        json_str = line[len("data: ") :]
        # Parse the JSON object
        data = json.loads(json_str.strip()) if json_str.strip() else None
        # Extract the content from the delta dictionary
        if data:
            if "choices" in data:
                if data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content


def call_provider_api_with_image(
    base64_image,
    api_key,
    base_url: str,
    model_name: str,
    question="Please perform OCR on this image",
    # stream: bool = True
):
    """
    Sends one image (base64) to the Albert API, along with a question prompt.
    Returns the response JSON.
    """
    url = base_url

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    model = model_name  #

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]

    data = {"model": model, "messages": messages, "temperature": 0.0}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # return process_response(response=response)
    if response.status_code == 200:

        return response.json()
    else:
        raise Exception(
            f"Request failed with status {response.status_code}: {response.text}"
        )


def analyse_image_provider(
    image: Any,  # File path string or Streamlit UploadedFile object
    api_key: str,
    base_url: str,
    model_name: str,
    question: str = "What is on this image?",
):
    """
    Analyze an image from either a file path or Streamlit uploaded file.

    Args:
        image: Either a file path (str) or Streamlit UploadedFile object
        api_key: API key for the vision service
        base_url: Base URL for the vision API
        model_name: Model name to use for analysis
        question: Question to ask about the image

    Returns:
        str: Analysis result from the vision API
    """
    # Handle both file paths and uploaded files
    if isinstance(image, str):
        # If it's a string, treat as file path
        pil_image = Image.open(image)
    else:
        # If it's an UploadedFile object, open directly
        pil_image = Image.open(image)

    # Convert to base64
    base64_image = image_to_base64(pil_image)

    # Call the API
    response_json = call_provider_api_with_image(
        base64_image=base64_image,
        api_key=api_key,
        question=question,
        base_url=base_url,
        model_name=model_name,
    )

    return response_json["choices"][0]["message"]["content"]


# Example Usage

# test_image = "path_to_image.png"
# response = analyse_image_provider(image=test_image, api_key=api_key, question="Perform OCR")
# print(response)
