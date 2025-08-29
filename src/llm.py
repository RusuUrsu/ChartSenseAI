'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain_community.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                'temperature': cfg.TEMPERATURE,
                                'context_length': 2048}
                        )

    return llm


def build_llm_for_chart_drawing():
    # Local CTransformers model
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                'temperature': 0.1,
                                'context_length': 2048}
                        )

    return llm


def build_agentic_llm():
    """Build LLM specifically optimized for agentic behavior with mathematical reasoning"""
    llm = CTransformers(model=cfg.MODEL_BIN_PATH,
                        model_type=cfg.MODEL_TYPE,
                        config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
                                'temperature': 0.2,  # Slightly higher for creative reasoning
                                'top_p': 0.9,
                                'top_k': 40,
                                'context_length': 2048}
                        )
    return llm


def invoke_llm_with_template(template_string, **kwargs):
    llm = build_llm_for_chart_drawing()

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=list(kwargs.keys()),
        template=template_string
    )

    # Format the prompt with provided variables
    formatted_prompt = prompt_template.format(**kwargs)

    # Get response from LLM
    response = llm.invoke(formatted_prompt)

    return response


def invoke_agentic_llm(template_string, **kwargs):
    """Invoke LLM with agentic capabilities for mathematical reasoning"""
    llm = build_agentic_llm()

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=list(kwargs.keys()),
        template=template_string
    )

    # Format the prompt with provided variables
    formatted_prompt = prompt_template.format(**kwargs)

    # Get response from LLM
    response = llm.invoke(formatted_prompt)

    return response
