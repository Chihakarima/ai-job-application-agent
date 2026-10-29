
import sys
import os
import time
import json

# ============================================================
# IMPORTS OUTILS
# ============================================================

from tools.web_tools import (
    web_search
)

from rag.retriever import (
    search_knowledge_base
)


# ============================================================
# 0. CHEMINS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    "history",
    "runs.json"
)

CONVERSATION_FILE = os.path.join(
    PROJECT_ROOT,
    "history",
    "conversation.json"
)


# ============================================================
# 1. MÉMOIRE CONVERSATIONNELLE
# ============================================================

def load_conversation():

    try:

        with open(
            CONVERSATION_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):

        return []


def save_conversation(
    user_message,
    assistant_message
):

    conversation = load_conversation()

    conversation.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    conversation.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    with open(
        CONVERSATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            conversation,
            f,
            indent=4,
            ensure_ascii=False
        )

    return conversation


def format_conversation(
    conversation
):

    if not conversation:

        return "Aucune conversation précédente."

    conversation_text = ""

    for message in conversation:

        role = message.get(
            "role",
            "unknown"
        )

        content = message.get(
            "content",
            ""
        )

        conversation_text += (
            f"{role}: {content}\n"
        )

    return conversation_text


# ============================================================
# IMPORTS LANGCHAIN / LANGGRAPH
# ============================================================

from typing import TypedDict

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    END
)

from langgraph.prebuilt import ToolNode

from agent.llm import llm

from tools.cv_tools import (
    get_cv_skills,
    get_job_offer,
    compare_skills,
    evaluate_application,
    get_improvement_advice
)


# ============================================================
# 2. OUTILS
# ============================================================

tools = [

    get_cv_skills,

    get_job_offer,

    compare_skills,

    evaluate_application,

    get_improvement_advice,

    web_search,

    search_knowledge_base
]


llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 3. MAPPING DES OUTILS
# ============================================================

TOOL_MAPPING = {

    "get_cv_skills":
        get_cv_skills,

    "get_job_offer":
        get_job_offer,

    "compare_skills":
        compare_skills,

    "evaluate_application":
        evaluate_application,

    "get_improvement_advice":
        get_improvement_advice,

    "web_search":
        web_search,

    "search_knowledge_base":
        search_knowledge_base
}


# ============================================================
# 4. STATE
# ============================================================

class State(TypedDict):

    user_question: str

    messages: list

    plan: str

    used_tools: list

    retry_count: int

    review: str

    review_reason: str

    errors: list

    timings: dict

    # AgentOps : LLM Monitoring

    llm_calls: int

    llm_model: str

    llm_time: float

    # AgentOps : Token Monitoring

    input_tokens: int

    output_tokens: int

    total_tokens: int

    # AgentOps : Tracing

    trace: list


# ============================================================
# 5. TOKEN MONITORING
# ============================================================

def get_token_usage(
    response
):

    usage = getattr(
        response,
        "usage_metadata",
        None
    )

    if not usage:

        return (
            0,
            0,
            0
        )

    input_tokens = usage.get(
        "input_tokens",
        0
    )

    output_tokens = usage.get(
        "output_tokens",
        0
    )

    total_tokens = usage.get(
        "total_tokens",
        input_tokens + output_tokens
    )

    return (
        input_tokens,
        output_tokens,
        total_tokens
    )


# ============================================================
# 6. HISTORY AGENTOPS
# ============================================================

def save_run_history(
    result,
    total_duration
):

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):

        history = []

    run = {

        "run_id":
            len(history) + 1,

        "duration":
            round(
                total_duration,
                2
            ),

        "llm_calls":
            result.get(
                "llm_calls",
                0
            ),

        "model":
            result.get(
                "llm_model",
                "qwen2.5:3b"
            ),

        "input_tokens":
            result.get(
                "input_tokens",
                0
            ),

        "output_tokens":
            result.get(
                "output_tokens",
                0
            ),

        "total_tokens":
            result.get(
                "total_tokens",
                0
            ),

        "tools":
            len(
                result.get(
                    "used_tools",
                    []
                )
            ),

        "retries":
            result.get(
                "retry_count",
                0
            ),

        "errors":
            len(
                result.get(
                    "errors",
                    []
                )
            )
    }

    history.append(
        run
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
            ensure_ascii=False
        )

    return run


# ============================================================
# 7. PLANNER
# ============================================================

def planner_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n🧠 PLANNER : création du plan..."
    )

    user_question = state[
        "user_question"
    ].lower()

    # ========================================================
    # Détection comparaison
    # ========================================================

    comparison_requested = any(

        keyword in user_question

        for keyword in [

            "compare",

            "comparaison",

            "compétences manquantes",

            "compétences manquante",

            "manquantes",

            "manquante",

            "correspondre"
        ]
    )

    # ========================================================
    # Détection évaluation
    # ========================================================

    evaluation_requested = any(

        keyword in user_question

        for keyword in [

            "évalue",

            "evaluation",

            "évaluation",

            "score",

            "compatibilité",

            "compatible",

            "note"
        ]
    )

    # ========================================================
    # Détection conseils
    # ========================================================

    advice_requested = any(

        keyword in user_question

        for keyword in [

            "conseil",

            "conseils",

            "conseiller",

            "améliorer",

            "amélioration",

            "améliore",

            "profil"
        ]
    )

    # ========================================================
    # Détection question générale RAG
    # ========================================================

    rag_requested = any(

        keyword in user_question

        for keyword in [

            "devenir data scientist",

            "data scientist",

            "data science",

            "que dois-je apprendre",

            "quelles compétences faut-il",

            "quelles compétences pour"
        ]
    )

    # ========================================================
    # Détection question personnalisée
    # ========================================================

    personalized_requested = any(

        keyword in user_question

        for keyword in [

            "mon cv",

            "mon profil",

            "mes compétences",

            "me manque",

            "moi",

            "pour moi"
        ]
    )

    # ========================================================
    # Construction du plan
    # ========================================================

    plan_steps = []

    # --------------------------------------------------------
    # Comparaison
    # --------------------------------------------------------

    if comparison_requested:

        plan_steps.extend(
            [
                "get_cv_skills",
                "get_job_offer",
                "compare_skills"
            ]
        )

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    if evaluation_requested:

        plan_steps.append(
            "evaluate_application"
        )

    # --------------------------------------------------------
    # Conseils
    # --------------------------------------------------------

    if advice_requested:

        plan_steps.append(
            "get_improvement_advice"
        )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if (
        rag_requested
        and not personalized_requested
    ):

        plan_steps.append(
            "search_knowledge_base"
        )

    # ========================================================
    # Debug
    # ========================================================

    print(
        f"📌 Question RAG : "
        f"{rag_requested}"
    )

    print(
        f"📌 Question personnalisée : "
        f"{personalized_requested}"
    )

    # ========================================================
    # Aucun outil détecté
    # ========================================================

    if not plan_steps:

        plan = (
            "Aucun outil spécifique détecté. "
            "L'Agent doit analyser la question."
        )

    else:

        plan = (
            "Plan pour répondre à la question actuelle :\n"
            +
            "\n".join(
                f"{i}. {tool}"
                for i, tool in enumerate(
                    plan_steps,
                    start=1
                )
            )
        )

    # ========================================================
    # Temps
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"⏱️ PLANNER : {elapsed:.2f} s"
    )

    # ========================================================
    # Trace
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "PLANNER"
    )

    # ========================================================
    # Retour
    # ========================================================

    return {

        "plan":
            plan,

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "planner":
                elapsed
        },

        "llm_calls":
            state.get(
                "llm_calls",
                0
            ),

        "llm_time":
            state.get(
                "llm_time",
                0.0
            ),

        "input_tokens":
            state.get(
                "input_tokens",
                0
            ),

        "output_tokens":
            state.get(
                "output_tokens",
                0
            ),

        "total_tokens":
            state.get(
                "total_tokens",
                0
            ),

        "trace":
            trace
    }


# ============================================================
# 8. EXTRACTION DES OUTILS DU PLAN
# ============================================================

def extract_tools_from_plan(
    plan
):

    planned_tools = []

    for tool_name in TOOL_MAPPING:

        if tool_name in plan:

            planned_tools.append(
                tool_name
            )

    return planned_tools


# ============================================================
# 9. AGENT
# ============================================================

def agent_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n🤖 AGENT : réflexion..."
    )

    user_question = state[
        "user_question"
    ]

    used_tools = state.get(
        "used_tools",
        []
    )

    plan = state.get(
        "plan",
        ""
    )

    # --------------------------------------------------------
    # Charger mémoire
    # --------------------------------------------------------

    conversation = load_conversation()

    conversation_text = format_conversation(
        conversation
    )

    # --------------------------------------------------------
    # Prompt système
    # --------------------------------------------------------

    system_prompt = """
Tu es un Agent IA spécialisé dans
l'analyse de candidatures.

Tu disposes des outils suivants :

- get_cv_skills
- get_job_offer
- compare_skills
- evaluate_application
- get_improvement_advice
- web_search
- search_knowledge_base


RÈGLES :

1. Utilise les outils nécessaires.

2. Suis le plan fourni par le Planner.

3. Si le Planner indique plusieurs outils,
ils doivent tous être exécutés.

4. Pour une comparaison :
   get_cv_skills
   get_job_offer
   compare_skills

5. Pour une évaluation :
   evaluate_application

6. Pour des conseils :
   get_improvement_advice

7. Pour une question générale sur
Data Scientist ou Data Science :
   search_knowledge_base

8. Pour une question générale,
n'utilise pas les informations
personnelles du CV pour répondre.

9. Pour une question personnalisée,
utilise les outils concernant le CV.

10. Ne calcule jamais toi-même le score.

11. Ne donne pas de score sans
evaluate_application.

12. Si le Reviewer demande un outil,
utilise cet outil.

13. Utilise l'historique seulement
comme contexte.

14. Pour une question générale,
utilise les informations retournées
par search_knowledge_base et
n'invente pas d'informations.
"""

    # --------------------------------------------------------
    # Contexte
    # --------------------------------------------------------

    context_prompt = f"""

QUESTION UTILISATEUR :

{user_question}


PLAN DU PLANNER :

{plan}


OUTILS DÉJÀ UTILISÉS :

{used_tools}


RETRIES :

{state.get("retry_count", 0)}


FEEDBACK DU REVIEWER :

{state.get("review_reason", "")}


HISTORIQUE :

{conversation_text}
"""

    messages = [

        SystemMessage(
            content=system_prompt
        ),

        HumanMessage(
            content=context_prompt
        )
    ]

    # --------------------------------------------------------
    # Appel LLM
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages
    )

    # --------------------------------------------------------
    # Outils demandés par le LLM
    # --------------------------------------------------------

    llm_tool_calls = []

    if hasattr(
        response,
        "tool_calls"
    ):

        llm_tool_calls = (
            response.tool_calls or []
        )

    # ========================================================
    # Compléter automatiquement les outils
    # manquants du plan
    # ========================================================

    planned_tools = extract_tools_from_plan(
        plan
    )

    planned_missing_tools = [

        tool_name

        for tool_name in planned_tools

        if tool_name not in used_tools
    ]

    requested_tool_names = [

        call.get(
            "name"
        )

        for call in llm_tool_calls
    ]

    final_tool_calls = list(
        llm_tool_calls
    )

    for tool_name in planned_missing_tools:

        if tool_name not in requested_tool_names:

            print(
                f"🔧 AGENT : ajout automatique "
                f"de {tool_name}"
            )

            final_tool_calls.append(

                {

                    "name":
                        tool_name,

                    "args":
                        {},

                    "id":
                        f"planned_{tool_name}"
                }
            )

    # --------------------------------------------------------
    # Recréer la réponse avec les tool calls
    # --------------------------------------------------------

    if final_tool_calls:

        response = AIMessage(

            content=
                response.content or "",

            tool_calls=
                final_tool_calls
        )

    # --------------------------------------------------------
    # Temps
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"⏱️ AGENT : {elapsed:.2f} s"
    )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    (
        input_tokens,
        output_tokens,
        total_tokens
    ) = get_token_usage(
        response
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "AGENT"
    )

    # --------------------------------------------------------
    # Retour
    # --------------------------------------------------------

    return {

        "messages": [
            response
        ],

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "agent":
                state.get(
                    "timings",
                    {}
                ).get(
                    "agent",
                    0
                ) + elapsed
        },

        "llm_calls":
            state.get(
                "llm_calls",
                0
            ) + 1,

        "llm_time":
            state.get(
                "llm_time",
                0.0
            ) + elapsed,

        "input_tokens":
            state.get(
                "input_tokens",
                0
            ) + input_tokens,

        "output_tokens":
            state.get(
                "output_tokens",
                0
            ) + output_tokens,

        "total_tokens":
            state.get(
                "total_tokens",
                0
            ) + total_tokens,

        "trace":
            trace
    }


# ============================================================
# 10. TOOLS
# ============================================================

def tools_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n🔧 TOOLS : exécution..."
    )

    tool_node = ToolNode(
        tools
    )

    result = tool_node.invoke(
        state
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"⏱️ TOOLS : {elapsed:.2f} s"
    )

    # --------------------------------------------------------
    # Outils utilisés
    # --------------------------------------------------------

    used_tools = state.get(
        "used_tools",
        []
    ).copy()

    for message in result[
        "messages"
    ]:

        if hasattr(
            message,
            "name"
        ):

            tool_name = message.name

            print(
                f"✅ TOOL EXÉCUTÉ : "
                f"{tool_name}"
            )

            if tool_name not in used_tools:

                used_tools.append(
                    tool_name
                )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "TOOLS"
    )

    return {

        "messages":
            result[
                "messages"
            ],

        "used_tools":
            used_tools,

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "tools":
                state.get(
                    "timings",
                    {}
                ).get(
                    "tools",
                    0
                ) + elapsed
        },

        "trace":
            trace
    }


# ============================================================
# 11. RETRY
# ============================================================

def retry_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n🔄 RETRY : correction du workflow..."
    )

    review_reason = state.get(
        "review_reason",
        ""
    )

    print(
        f"📌 Reviewer : "
        f"{review_reason}"
    )

    selected_tool = None

    # --------------------------------------------------------
    # Chercher le premier outil manquant
    # --------------------------------------------------------

    for tool_name in TOOL_MAPPING:

        if tool_name in review_reason:

            selected_tool = tool_name

            break

    # --------------------------------------------------------
    # Exécuter le retry
    # --------------------------------------------------------

    if selected_tool:

        print(
            f"🔧 RETRY TOOL : "
            f"{selected_tool}"
        )

        retry_message = AIMessage(

            content="",

            tool_calls=[

                {

                    "name":
                        selected_tool,

                    "args":
                        {},

                    "id":
                        f"retry_{selected_tool}"
                }

            ]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        trace = state.get(
            "trace",
            []
        ).copy()

        trace.append(
            "RETRY"
        )

        return {

            "messages": [
                retry_message
            ],

            "retry_count":
                state.get(
                    "retry_count",
                    0
                ) + 1,

            "timings": {

                **state.get(
                    "timings",
                    {}
                ),

                "retry":
                    state.get(
                        "timings",
                        {}
                    ).get(
                        "retry",
                        0
                    ) + elapsed
            },

            "trace":
                trace
        }

    # --------------------------------------------------------
    # Aucun outil identifié
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "RETRY"
    )

    return {

        "retry_count":
            state.get(
                "retry_count",
                0
            ) + 1,

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "retry":
                state.get(
                    "timings",
                    {}
                ).get(
                    "retry",
                    0
                ) + elapsed
        },

        "trace":
            trace
    }


# ============================================================
# 12. AGENT ROUTER
# ============================================================

def agent_router(
    state: State
):

    last_message = state[
        "messages"
    ][-1]

    # --------------------------------------------------------
    # Si l'Agent a des tool calls
    # --------------------------------------------------------

    if hasattr(
        last_message,
        "tool_calls"
    ):

        if last_message.tool_calls:

            return "tools"

    # --------------------------------------------------------
    # Sinon Reviewer
    # --------------------------------------------------------

    return "reviewer"


# ============================================================
# 13. REVIEWER
# ============================================================

def reviewer_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n🔎 REVIEWER : vérification..."
    )

    user_question = state[
        "user_question"
    ].lower()

    used_tools = state.get(
        "used_tools",
        []
    )

    # ========================================================
    # Détection comparaison
    # ========================================================

    comparison_requested = any(

        keyword in user_question

        for keyword in [

            "compare",

            "comparaison",

            "compétences manquantes",

            "compétences manquante",

            "manquantes",

            "manquante",

            "correspondre"
        ]
    )

    # ========================================================
    # Détection évaluation
    # ========================================================

    evaluation_requested = any(

        keyword in user_question

        for keyword in [

            "évalue",

            "evaluation",

            "évaluation",

            "score",

            "compatibilité",

            "compatible",

            "note"
        ]
    )

    # ========================================================
    # Détection conseils
    # ========================================================

    advice_requested = any(

        keyword in user_question

        for keyword in [

            "conseil",

            "conseils",

            "conseiller",

            "améliorer",

            "amélioration",

            "améliore",

            "profil"
        ]
    )

    # ========================================================
    # Détection RAG
    # ========================================================

    rag_requested = any(

        keyword in user_question

        for keyword in [

            "devenir data scientist",

            "data scientist",

            "data science",

            "que dois-je apprendre",

            "quelles compétences faut-il",

            "quelles compétences pour"
        ]
    )

    # ========================================================
    # Détection question personnalisée
    # ========================================================

    personalized_requested = any(

        keyword in user_question

        for keyword in [

            "mon cv",

            "mon profil",

            "mes compétences",

            "me manque",

            "moi",

            "pour moi"
        ]
    )

    # ========================================================
    # Debug
    # ========================================================

    print(
        f"📌 Comparaison demandée : "
        f"{comparison_requested}"
    )

    print(
        f"📌 Évaluation demandée : "
        f"{evaluation_requested}"
    )

    print(
        f"📌 Conseils demandés : "
        f"{advice_requested}"
    )

    print(
        f"📌 RAG demandé : "
        f"{rag_requested}"
    )

    print(
        f"📌 Question personnalisée : "
        f"{personalized_requested}"
    )

    print(
        f"📌 Outils utilisés : "
        f"{used_tools}"
    )

    # ========================================================
    # Outils manquants
    # ========================================================

    missing_tools = []

    # --------------------------------------------------------
    # Comparaison
    # --------------------------------------------------------

    if comparison_requested:

        required_tools = [

            "get_cv_skills",

            "get_job_offer",

            "compare_skills"
        ]

        for tool in required_tools:

            if tool not in used_tools:

                missing_tools.append(
                    tool
                )

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    if evaluation_requested:

        if (
            "evaluate_application"
            not in used_tools
        ):

            missing_tools.append(
                "evaluate_application"
            )

    # --------------------------------------------------------
    # Conseils
    # --------------------------------------------------------

    if advice_requested:

        if (
            "get_improvement_advice"
            not in used_tools
        ):

            missing_tools.append(
                "get_improvement_advice"
            )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if (
        rag_requested
        and not personalized_requested
    ):

        if (
            "search_knowledge_base"
            not in used_tools
        ):

            missing_tools.append(
                "search_knowledge_base"
            )

    # ========================================================
    # Temps + trace
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start
    )

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "REVIEWER"
    )

    # ========================================================
    # CONTINUE
    # ========================================================

    if missing_tools:

        reason = (
            "Outils manquants : "
            +
            ", ".join(
                missing_tools
            )
        )

        print(
            f"⚠️ REVIEWER : "
            f"{reason}"
        )

        return {

            "review":
                "CONTINUE",

            "review_reason":
                reason,

            "timings": {

                **state.get(
                    "timings",
                    {}
                ),

                "reviewer":
                    elapsed
            },

            "trace":
                trace
        }

    # ========================================================
    # DONE
    # ========================================================

    print(
        "🔎 REVIEWER : DONE"
    )

    return {

        "review":
            "DONE",

        "review_reason":
            "Tous les outils nécessaires ont été exécutés.",

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "reviewer":
                elapsed
        },

        "trace":
            trace
    }


# ============================================================
# 14. REVIEW ROUTER
# ============================================================

def review_router(
    state: State
):

    if state.get(
        "review"
    ) == "DONE":

        return "final_answer"

    if state.get(
        "review"
    ) == "CONTINUE":

        return "agent_retry"

    return "agent"


# ============================================================
# 15. FINAL ANSWER
# ============================================================

def final_answer_node(
    state: State
):

    start = time.perf_counter()

    print(
        "\n📝 FINAL ANSWER : génération..."
    )

    user_question = state[
        "user_question"
    ]

    messages = state[
        "messages"
    ]

    # --------------------------------------------------------
    # Mémoire
    #
    # IMPORTANT :
    # Pour une question générale RAG,
    # on n'utilise PAS l'historique.
    # --------------------------------------------------------

    user_question_lower = user_question.lower()

    rag_requested = any(
        keyword in user_question_lower
        for keyword in [
            "devenir data scientist",
            "data scientist",
            "data science",
            "que dois-je apprendre",
            "quelles compétences faut-il",
            "quelles compétences pour"
        ]
    )

    personalized_requested = any(
        keyword in user_question_lower
        for keyword in [
            "mon cv",
            "mon profil",
            "mes compétences",
            "me manque",
            "moi",
            "pour moi"
        ]
    )

    if (
        rag_requested
        and not personalized_requested
    ):

        conversation_text = (
            "Pas d'historique utilisé pour "
            "cette question générale."
        )

    else:

        conversation = load_conversation()

        conversation_text = format_conversation(
            conversation
        )

    # --------------------------------------------------------
    # Prompt final
    # --------------------------------------------------------

    prompt = f"""
Tu es l'assistant final d'un Agent IA
d'analyse de candidatures.

QUESTION ACTUELLE :

{user_question}


HISTORIQUE :

{conversation_text}


RÉSULTATS DES OUTILS :

{messages}


RÈGLES :

1. Réponds simplement et clairement.

2. Utilise uniquement les résultats
disponibles dans les outils.

3. Si la question demande une comparaison,
indique les compétences manquantes.

4. Si la question demande des conseils,
utilise les résultats de
get_improvement_advice.

5. Si la question demande une évaluation,
utilise le score de evaluate_application.

6. Si la question est générale sur
Data Scientist ou Data Science,
utilise uniquement les informations
retournées par search_knowledge_base.

7. Pour une question générale,
ne transforme pas la réponse en
évaluation personnalisée du CV.

8. Si la question est personnalisée,
tu peux utiliser les informations
du CV et des outils correspondants.

9. Ne calcule jamais toi-même un score.

10. N'invente aucune information.

11. Ne prétends jamais que la candidature
a été acceptée ou que l'utilisateur
a été recruté.
"""

    # --------------------------------------------------------
    # Appel LLM
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"⏱️ FINAL ANSWER : "
        f"{elapsed:.2f} s"
    )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    (
        input_tokens,
        output_tokens,
        total_tokens
    ) = get_token_usage(
        response
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "FINAL_ANSWER"
    )

    # --------------------------------------------------------
    # Retour
    # --------------------------------------------------------

    return {

        "messages": [
            response
        ],

        "timings": {

            **state.get(
                "timings",
                {}
            ),

            "final_answer":
                elapsed
        },

        "llm_calls":
            state.get(
                "llm_calls",
                0
            ) + 1,

        "llm_time":
            state.get(
                "llm_time",
                0.0
            ) + elapsed,

        "input_tokens":
            state.get(
                "input_tokens",
                0
            ) + input_tokens,

        "output_tokens":
            state.get(
                "output_tokens",
                0
            ) + output_tokens,

        "total_tokens":
            state.get(
                "total_tokens",
                0
            ) + total_tokens,

        "trace":
            trace
    }


# ============================================================
# 16. CONSTRUCTION DU GRAPH
# ============================================================

graph_builder = StateGraph(
    State
)


# ============================================================
# NODES
# ============================================================

graph_builder.add_node(
    "planner",
    planner_node
)

graph_builder.add_node(
    "agent",
    agent_node
)

graph_builder.add_node(
    "tools",
    tools_node
)

graph_builder.add_node(
    "reviewer",
    reviewer_node
)

graph_builder.add_node(
    "agent_retry",
    retry_node
)

graph_builder.add_node(
    "final_answer",
    final_answer_node
)


# ============================================================
# START
# ============================================================

graph_builder.set_entry_point(
    "planner"
)


# ============================================================
# PLANNER → AGENT
# ============================================================

graph_builder.add_edge(
    "planner",
    "agent"
)


# ============================================================
# AGENT → TOOLS / REVIEWER
# ============================================================

graph_builder.add_conditional_edges(

    "agent",

    agent_router,

    {

        "tools":
            "tools",

        "reviewer":
            "reviewer"
    }
)


# ============================================================
# TOOLS → REVIEWER
# ============================================================

graph_builder.add_edge(
    "tools",
    "reviewer"
)


# ============================================================
# REVIEWER → RETRY / FINAL
# ============================================================

graph_builder.add_conditional_edges(

    "reviewer",

    review_router,

    {

        "agent_retry":
            "agent_retry",

        "agent":
            "agent",

        "final_answer":
            "final_answer"
    }
)


# ============================================================
# RETRY → TOOLS
# ============================================================

graph_builder.add_edge(
    "agent_retry",
    "tools"
)


# ============================================================
# FINAL → END
# ============================================================

graph_builder.add_edge(
    "final_answer",
    END
)


# ============================================================
# COMPILE
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 17. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "🤖 AI JOB APPLICATION AGENT"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Mémoire
    # --------------------------------------------------------

    conversation_history = load_conversation()

    print(
        f"\n🧠 Mémoire : "
        f"{len(conversation_history)} "
        f"messages précédents"
    )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = input(
        "\n💬 Votre question : "
    )

    run_start = time.perf_counter()

    # --------------------------------------------------------
    # State initial
    # --------------------------------------------------------

    initial_state = {

        "user_question":
            question,

        "messages": [

            HumanMessage(
                content=question
            )

        ],

        "plan":
            "",

        "used_tools":
            [],

        "retry_count":
            0,

        "review":
            "",

        "review_reason":
            "",

        "errors":
            [],

        "timings":
            {},

        # AgentOps

        "llm_calls":
            0,

        "llm_model":
            "qwen2.5:3b",

        "llm_time":
            0.0,

        # Tokens

        "input_tokens":
            0,

        "output_tokens":
            0,

        "total_tokens":
            0,

        # Tracing

        "trace": [

            "START"

        ]
    }

    # --------------------------------------------------------
    # Exécution
    # --------------------------------------------------------

    try:

        result = graph.invoke(
            initial_state
        )

    except Exception as e:

        print(
            "\n❌ ERREUR :",
            e
        )

        errors = initial_state.get(
            "errors",
            []
        ).copy()

        errors.append(
            str(e)
        )

        result = {

            **initial_state,

            "errors":
                errors
        }

    # --------------------------------------------------------
    # Temps total
    # --------------------------------------------------------

    run_duration = (
        time.perf_counter()
        - run_start
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = result.get(
        "trace",
        []
    ).copy()

    if (
        not trace
        or
        trace[-1] != "END"
    ):

        trace.append(
            "END"
        )

    print(
        "\n🔎 TRACE"
    )

    print(
        " → ".join(
            trace
        )
    )

    # ========================================================
    # AGENTOPS
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "📊 AGENTOPS"
    )

    print(
        "======================================"
    )

    print(
        f"⏱️ Temps total     : "
        f"{run_duration:.2f} s"
    )

    print(
        f"📞 Appels LLM     : "
        f"{result.get('llm_calls', 0)}"
    )

    print(
        f"🤖 Modèle LLM     : "
        f"{result.get('llm_model', 'qwen2.5:3b')}"
    )

    print(
        f"⏱️ Temps LLM      : "
        f"{result.get('llm_time', 0.0):.2f} s"
    )

    print(
        f"📥 Input tokens   : "
        f"{result.get('input_tokens', 0)}"
    )

    print(
        f"📤 Output tokens  : "
        f"{result.get('output_tokens', 0)}"
    )

    print(
        f"📊 Total tokens   : "
        f"{result.get('total_tokens', 0)}"
    )

    print(
        f"🔧 Outils utilisés : "
        f"{len(result.get('used_tools', []))}"
    )

    print(
        f"🔄 Retries        : "
        f"{result.get('retry_count', 0)}"
    )

    print(
        f"❌ Erreurs        : "
        f"{len(result.get('errors', []))}"
    )

    # ========================================================
    # OUTILS
    # ========================================================

    print(
        "\n🔧 OUTILS UTILISÉS :"
    )

    for tool in result.get(
        "used_tools",
        []
    ):

        print(
            f"  - {tool}"
        )

    # ========================================================
    # RÉPONSE FINALE
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "💬 RÉPONSE FINALE"
    )

    print(
        "======================================"
    )

    final_message = result[
        "messages"
    ][-1]

    final_answer = final_message.content

    print(
        final_answer
    )

    # ========================================================
    # MÉMOIRE
    # ========================================================

    save_conversation(
        question,
        final_answer
    )

    print(
        "\n🧠 Conversation sauvegardée"
    )

    print(
        "📁 Fichier : "
        "history/conversation.json"
    )

    # ========================================================
    # HISTORY AGENTOPS
    # ========================================================

    saved_run = save_run_history(
        result,
        run_duration
    )

    print(
        "\n======================================"
    )

    print(
        f"✅ Run #{saved_run['run_id']} sauvegardé"
    )

    print(
        "📁 Fichier : history/runs.json"
    )

    print(
        "======================================"
    )
