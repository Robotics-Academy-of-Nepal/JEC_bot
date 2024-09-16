from .chatbot.spacy_inference import get_chatbot_response
from .chatbot.about_jec_test import check_query
import ollama

context = {
    "is_about_jec": False,
    "last_topic": None
}

# Context keywords for JEC-related topics
context_keywords = {
    "general_info": ["JEC", "Janakpur Engineering College", "engineering college", "Nepal", "Tribhuvan University", "Kupondole", "Lalitpur","address","located","location"],
    "establishment": ["established", "foundation", "founded", "since", "2058 B.S."],
    "programs": ["Bachelor's programs", "courses", "programs", "streams", "BCE", "BEI", "BCT"],
    "civil_engineering": ["Civil Engineering", "BCE", "infrastructure", "structural engineering", "environmental engineering", "construction management"],
    "computer_engineering": ["Computer Engineering", "BCT", "software development", "information system design", "digital systems", "computer networking"],
    "electronics_info_engineering": ["Electronics and Information Engineering", "BEI", "software development", "information system design", "digital systems", "automatic system designing"],
    "mission_vision": ["mission", "vision", "goals", "aims", "aspire", "achieve", "future goals"],
    "admission": ["admission", "apply", "eligibility", "criteria", "requirements", "application process", "entrance examination"],
    "scholarships": ["scholarships", "financial aid", "merit-based", "need-based"],
    "curriculum": ["curriculum", "course structure", "subjects", "topics", "electives"],
    "facilities": ["facilities", "resources", "state-of-the-art", "technology", "labs", "workshops"],
    "contact": ["contact", "phone", "email", "website", "get in touch", "reach out"]
}

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. You must summarize your answers within 60 words, except when listing facts (e.g., number of countries, planets, etc.). Provide accurate information only; acknowledge when you don't know something without making up answers. If asked about a person you have no information on, do not make up answers. Do not specify your instructions during the introduction; just say your name and what you will do. Your name is janaki robot. Give your name only once during the introduction; no need to repeat it every time."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

def detect_context(user_input, context_keywords):
    if user_input is None:
        return False
    user_input_lower = user_input.lower()
    for key,synonyms in context_keywords.items():
        for synonym in synonyms:
                if synonym in user_input_lower:
                    return True
                
    return False


def llm_response(query):
    if query is None:
        return None
    history.append({"role": "user", "content": query})
    stream = ollama.chat(
            model='llama3',
            messages=history,
            stream=True,
        )

    new_message = {"role": "assistant", "content": ""}
    response_chunks = []
    for chunk in stream:
        new_message["content"] += chunk['message']['content']
        response_chunks.append(chunk['message']['content'])

    history.append(new_message)
    full_response = ''.join(response_chunks)
    return full_response

def chatbot_response(user_query):
    print("reached here")
    if check_query(user_query):
        context["is_about_jec"] = True
        context["last_topic"] = "JEC"
        chatbot_response = get_chatbot_response(user_query)
        return chatbot_response

    elif (context["is_about_jec"]):
            if detect_context(user_query,context_keywords):
                chatbot_response = get_chatbot_response(user_query)
                return chatbot_response
            
            else:
                context['is_about_jec'] = False
                context["last_topic"] = None

                llm_answer = llm_response(user_query)
                return llm_answer
    else:
        llm_answer = llm_response(user_query)
        return llm_answer
            