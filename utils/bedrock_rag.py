import boto3
import json
import os


def generate_answer(question, text_results, image_results):

    # Create context from retrieved text
    text_context = "\n".join(text_results)

    # Create context from image captions
    image_context = "\n".join(image_results)

    # Combine both
    context = f"""
TEXT INFORMATION:
{text_context}

IMAGE INFORMATION:
{image_context}
"""

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the information retrieved
from the document.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Give a clear and simple answer.
If the information is not available in the retrieved context,
say that the information is not available.
"""

    # AWS Bedrock client
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 500,
            "temperature": 0.2
        }
    }

    response = client.invoke_model(
        modelId=os.getenv(
            "BEDROCK_MODEL_ID",
            "amazon.nova-lite-v1:0"
        ),
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())

    return result