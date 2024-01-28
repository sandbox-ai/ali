from src.rag_session import RAGSession
import src.custom_logging as logger
import logging
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display


# Set env vars
load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
# Set up logging level:
logging.basicConfig(level=logging.INFO)

# Path to session configuration file:
config_filepath = r"./config.json"
config_filepath
# Set up RAG session:
session = RAGSession(config_filepath)
session.set_up()
dir(session)

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

# Display the prompts used now
dir(session.query_engine)
prompts_dict = session.query_engine.get_prompts()
display_prompt_dict(prompts_dict)

# explore the prompts one by one
prompts_dict.keys()



# text_qa_template prompt
text_qa_prompt = prompts_dict['response_synthesizer:text_qa_template']
# text_qa_template prompt
refine_prompt = prompts_dict['response_synthesizer:refine_template']


# Display the prompts
def display_prompt_template(prompt_template):
    print(f"Metadata: {prompt_template.metadata}")
    print(f"Template Variables: {prompt_template.template_vars}")
    print(f"Default Template: {prompt_template.default_template.template}")

    for i, conditional in enumerate(prompt_template.conditionals):
        condition, template = conditional
        print(f"\nConditional {i+1}:")
        print(f"Condition Function: {condition.__name__}")
        if hasattr(template, 'template'):
            print(f"Template: {template.template}")
        if hasattr(template, 'message_templates'):
            for j, message in enumerate(template.message_templates):
                print(f"Message {j+1}:")
                print(f"Role: {message.role}")
                print(f"Content: {message.content}")


display_prompt_template(text_qa_prompt)
display_prompt_template(refine_prompt)

# Are both prompts the same?
if text_qa_prompt == refine_prompt:
    print("text_qa_prompt and refine_prompt are the same.")
else:
    print("text_qa_prompt and refine_prompt are NOT the same.")




# Extract content of prompts
def extract_content(prompt_template):
    contents = []
    # Extract the template from the default_template
    contents.append(prompt_template.default_template.template)
    
    # Extract the content from the conditionals
    for conditional in prompt_template.conditionals:
        _, template = conditional
        if hasattr(template, 'message_templates'):
            for message in template.message_templates:
                contents.append(message.content)
    return contents


content_text_qa_prompt = extract_content(text_qa_prompt)
content_refine_prompt = extract_content(refine_prompt)

content_text_qa_prompt
content_refine_prompt
content_text_qa_prompt == content_refine_prompt




