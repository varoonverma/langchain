import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings

from config import LLM_MODEL, LLM_TEMPERATURE, LLAMA_MODEL_PATH, LLAMA_MODEL_N_CTX


class ModelFactory:
    @staticmethod
    def get_llm(api_key=None, use_openai=True):
        if use_openai:
            if not api_key:
                return None

            os.environ["OPENAI_API_KEY"] = api_key
            return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        else:
            try:
                return LlamaCpp(model_path=LLAMA_MODEL_PATH, n_ctx=LLAMA_MODEL_N_CTX)
            except Exception:
                return None

    @staticmethod
    def get_embeddings(api_key=None, use_openai=True):
        if use_openai:
            if not api_key:
                return None

            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAIEmbeddings()
        else:
            try:
                return LlamaCppEmbeddings(model_path=LLAMA_MODEL_PATH)
            except Exception:
                return None


def generate_answer(llm, query, flight_data):
    if not llm:
        return "Error: LLM model could not be initialized"

    flight_table = "\n".join([
        f"{row[0]} {row[1]} on {row[2]}: {row[3]} ({row[4]}) to {row[5]} ({row[6]})"
        for row in flight_data
    ])

    prompt = "Based on the following flight data, answer this question: " + query + "\n\n"
    prompt += "Flight data:\n" + flight_table + "\n\n"
    prompt += "If you can't answer based on the data, say \"I don't have enough information about that.\""

    try:
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else response
            return content
        else:
            return llm(prompt)
    except Exception as e:
        return f"Error generating answer: {str(e)}"