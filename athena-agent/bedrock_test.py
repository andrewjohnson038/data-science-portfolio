
import boto3
import config
import json


# Initialize Bedrock client
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=config.AWS_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
)

llama3_model = "meta.llama3-70b-instruct-v1:0"

try:
    # Attempt to list models (may fail if unsupported)
    response = bedrock.list_foundation_models()
    model_ids = [model['modelId'] for model in response['modelSummaries']]
    print("✅ Bedrock is reachable. Available models:")
    for model_id in model_ids:
        print(f" - {model_id}")

    if llama3_model in model_ids:
        print(f"\n✅ You have access to: {llama3_model}")
    else:
        print(f"\n⚠️ You do NOT have access to: {llama3_model} (Request access in AWS Bedrock Console)")

except AttributeError:
    # list_foundation_models not supported, so try invoking directly
    print("⚠️ list_foundation_models() not supported by boto3 client. Skipping listing models.")

try:
    response = bedrock.invoke_model(
        modelId=llama3_model,
        body=json.dumps({
            "prompt": config.SYSTEM_PROMPT,
            "max_gen_len": 10,
            "temperature": 0.3,
            "top_p": 0.95
        }),
        contentType='application/json'
    )

    response_body = json.loads(response['body'].read())
    print("✅ Successfully invoked model!")
    print(f"Response: {response_body}")

except Exception as e:
    print(f"❌ Error invoking model: {e}")


# temperature controls the randomness of the model's output.
# Values closer to 0 make the output more deterministic and focused,
# often repeating the most likely next words.
# Values closer to 1 increase creativity and variety,
# allowing the model to pick less likely options.
# Example: temperature=0.1 produces predictable, conservative responses,
# temperature=0.9 produces more diverse and creative responses.

# top_p is the nucleus sampling parameter.
# It restricts the model to sample from the smallest set of words
# whose cumulative probability is at least top_p.
# For example, top_p=0.9 means the model only considers words
# that make up 90% of the probability mass.
# This filters out very unlikely words, keeping output coherent but varied.
