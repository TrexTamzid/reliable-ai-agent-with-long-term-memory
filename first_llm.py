from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")

conversation = []

while True:

    prompt = input("You: ")

    # Stop the program when the user types "exit"
    if prompt.lower() == "exit":
        break

    # Add user's message to conversation memory
    conversation.append({"role": "user", "content": prompt})

    # Send the complete conversation to the model
    messages = conversation
    text = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)

    # Convert text into tokens
    inputs = tokenizer( text, return_tensors="pt")

    # Generate response
    outputs = model.generate( **inputs, max_new_tokens=150)

    # Get only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    # Convert generated tokens back into text
    response = tokenizer.decode( generated_ids, skip_special_tokens=True)

    # Add AI response to conversation memory
    conversation.append({"role": "assistant", "content": response})

    # Display only the AI's response
    print("\nAI:")
    print(response)

print("\nConversation ended.")