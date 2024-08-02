from transformers import pipeline
import chainlit as cl

# Initialize the text-generation pipeline from Hugging Face
pipe = pipeline("text-generation", model="gpt2")

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Access the message content
        user_input = message.content
        
        # Generate a response with a refined prompt and parameters
        prompt = f"Respond to the following message in a conversational and engaging manner: {user_input}"
        response = pipe(prompt, max_length=100, num_return_sequences=1, temperature=0.8)
        
        # Extract the generated text, making sure to remove the prompt part
        generated_text = response[0]['generated_text']
        response_text = generated_text[len(prompt):].strip()
        
        # Send the generated text as the response
        await cl.Message(content=response_text).send()
    except Exception as e:
        # Handle any errors that occur
        await cl.Message(content=f"An error occurred: {str(e)}").send()
