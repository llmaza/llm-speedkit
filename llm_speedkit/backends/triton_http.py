# Use this if you want classic Triton model inference endpoint first.

# Need config:

# server_url

# triton_model_name

# And then request payload will be Triton-specific.

# This is a bit more infra-ish, a bit less convenient for unified LLM comparison.

# My recommendation

# Start with:

# triton_openai first

# because it aligns better with:

# unified client harness

# vLLM comparison

# future GenAI-Perf style evaluation