import time
import json
import os
import streamlit as st

from agent.graph import graph, save_run_history
from tools.cv_tools import set_current_cv


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CHEMIN HISTORIQUE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    "history",
    "runs.json"
)


# ============================================================
# FONCTION : CHARGER L'HISTORIQUE
# ============================================================

def load_run_history():

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        FileNotFoundError,
        json.JSONDecodeError
    ):

        return []


# ============================================================
# TITRE
# ============================================================

st.title(
    "🤖 AI Job Application Agent"
)

st.write(
    "Agent intelligent pour analyser un CV, "
    "une offre d'emploi et évaluer une candidature."
)


# ============================================================
# UPLOAD DU CV
# ============================================================

st.subheader("📄 Votre CV")

uploaded_cv = st.file_uploader(
    "Déposez votre CV",
    type=["txt", "pdf"]
)


# ============================================================
# LECTURE DU CV
# ============================================================

cv_text = ""

if uploaded_cv is not None:

    st.success(
        f"✅ CV chargé : {uploaded_cv.name}"
    )

    try:

        # ----------------------------------------------------
        # CV TXT
        # ----------------------------------------------------

        if uploaded_cv.name.lower().endswith(".txt"):

            cv_text = uploaded_cv.read().decode(
                "utf-8"
            )


        # ----------------------------------------------------
        # CV PDF
        # ----------------------------------------------------

        elif uploaded_cv.name.lower().endswith(".pdf"):

            from pypdf import PdfReader

            reader = PdfReader(
                uploaded_cv
            )

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    cv_text += text + "\n"


        # ----------------------------------------------------
        # STOCKAGE DU CV POUR LA SESSION
        # ----------------------------------------------------

        if cv_text.strip():

            set_current_cv(
                cv_text
            )

            st.success(
                "✅ Contenu du CV chargé dans l'Agent."
            )

        else:

            st.warning(
                "⚠️ Le contenu du CV n'a pas pu être extrait."
            )


    except Exception as e:

        st.error(
            "❌ Impossible de lire le CV."
        )

        st.exception(e)


# ============================================================
# QUESTION
# ============================================================

question = st.text_area(
    "💬 Votre question",

    placeholder=(
        "Exemple : Compare mon CV avec l'offre, "
        "indique mes compétences manquantes "
        "et donne-moi des conseils."
    ),

    height=120
)


# ============================================================
# BOUTON
# ============================================================

if st.button(
    "🚀 Analyser",
    type="primary"
):

    if not question.strip():

        st.warning(
            "⚠️ Veuillez entrer une question."
        )

    elif uploaded_cv is None:

        st.warning(
            "⚠️ Veuillez d'abord déposer votre CV."
        )

    elif not cv_text.strip():

        st.warning(
            "⚠️ Le contenu du CV est vide ou n'a pas pu être extrait."
        )

    else:

        with st.spinner(
            "🤖 L'agent travaille..."
        ):

            try:

                # =================================================
                # CHRONOMÈTRE
                # =================================================

                start_time = time.perf_counter()


                # =================================================
                # STATE INITIAL
                # =================================================

                initial_state = {

                    "user_question":
                        question,

                    "messages": [],

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

                    "llm_calls":
                        0,

                    "llm_model":
                        "qwen2.5:3b",

                    "llm_time":
                        0.0,

                    "input_tokens":
                        0,

                    "output_tokens":
                        0,

                    "total_tokens":
                        0,

                    "trace":
                        [
                            "START"
                        ]
                }


                # =================================================
                # EXÉCUTION DU GRAPH
                # =================================================

                result = graph.invoke(
                    initial_state
                )


                # =================================================
                # TEMPS TOTAL
                # =================================================

                end_time = time.perf_counter()

                latency = (
                    end_time - start_time
                )


                # =================================================
                # SAUVEGARDE AGENTOPS
                # =================================================

                save_run_history(
                    result,
                    latency
                )


                # =================================================
                # RÉCUPÉRATION DES DONNÉES
                # =================================================

                plan = result.get(
                    "plan",
                    ""
                )

                used_tools = result.get(
                    "used_tools",
                    []
                )

                review = result.get(
                    "review",
                    ""
                )

                review_reason = result.get(
                    "review_reason",
                    ""
                )

                retry_count = result.get(
                    "retry_count",
                    0
                )

                errors = result.get(
                    "errors",
                    []
                )

                llm_calls = result.get(
                    "llm_calls",
                    0
                )

                llm_model = result.get(
                    "llm_model",
                    "qwen2.5:3b"
                )

                llm_time = result.get(
                    "llm_time",
                    0.0
                )

                input_tokens = result.get(
                    "input_tokens",
                    0
                )

                output_tokens = result.get(
                    "output_tokens",
                    0
                )

                total_tokens = result.get(
                    "total_tokens",
                    0
                )

                trace = result.get(
                    "trace",
                    []
                )


                # =================================================
                # RÉPONSE FINALE
                # =================================================

                final_message = result[
                    "messages"
                ][-1]

                final_answer = (
                    final_message.content
                )


                # =================================================
                # TITRE AGENTOPS
                # =================================================

                st.divider()

                st.header(
                    "📊 AgentOps"
                )


                # =================================================
                # MÉTRIQUES PRINCIPALES
                # =================================================

                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "⏱️ Temps total",
                        f"{latency:.2f} s"
                    )


                with col2:

                    st.metric(
                        "🤖 Appels LLM",
                        llm_calls
                    )


                with col3:

                    st.metric(
                        "🎟️ Total tokens",
                        total_tokens
                    )


                with col4:

                    st.metric(
                        "🔄 Retries",
                        retry_count
                    )


                # =================================================
                # INFORMATIONS LLM
                # =================================================

                st.subheader(
                    "🤖 LLM Monitoring"
                )

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.write(
                        "**Modèle**"
                    )

                    st.info(
                        llm_model
                    )


                with col2:

                    st.write(
                        "**Temps LLM**"
                    )

                    st.info(
                        f"{llm_time:.2f} s"
                    )


                with col3:

                    st.write(
                        "**Appels LLM**"
                    )

                    st.info(
                        str(llm_calls)
                    )


                # =================================================
                # TOKEN MONITORING
                # =================================================

                st.subheader(
                    "🎟️ Token Monitoring"
                )

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "📥 Input",
                        input_tokens
                    )


                with col2:

                    st.metric(
                        "📤 Output",
                        output_tokens
                    )


                with col3:

                    st.metric(
                        "📊 Total",
                        total_tokens
                    )


                # =================================================
                # TOOLS
                # =================================================

                st.subheader(
                    "🔧 Tools utilisés"
                )


                if used_tools:

                    for tool in used_tools:

                        st.success(
                            f"✓ {tool}"
                        )

                else:

                    st.info(
                        "Aucun outil utilisé."
                    )


                # =================================================
                # TRACE
                # =================================================

                st.subheader(
                    "🔎 Tracing"
                )


                if trace:

                    st.code(
                        " → ".join(
                            trace
                        )
                    )

                else:

                    st.info(
                        "Aucune trace disponible."
                    )


                # =================================================
                # REVIEWER
                # =================================================

                st.subheader(
                    "🔎 Reviewer"
                )


                if review == "DONE":

                    st.success(
                        "✅ Workflow validé par le Reviewer."
                    )

                elif review == "CONTINUE":

                    st.warning(
                        "⚠️ Le Reviewer demande "
                        "une correction."
                    )

                else:

                    st.info(
                        "ℹ️ Statut du Reviewer non disponible."
                    )


                if review_reason:

                    with st.expander(
                        "Voir le raisonnement du Reviewer"
                    ):

                        st.write(
                            review_reason
                        )


                # =================================================
                # ERREURS
                # =================================================

                st.subheader(
                    "❌ Erreurs"
                )


                if errors:

                    for error in errors:

                        st.error(
                            error
                        )

                else:

                    st.success(
                        "Aucune erreur."
                    )


                # =================================================
                # PLAN
                # =================================================

                st.subheader(
                    "📋 Plan de l'Agent"
                )


                if plan:

                    st.write(
                        plan
                    )

                else:

                    st.info(
                        "Aucun plan disponible."
                    )


                # =================================================
                # RÉPONSE FINALE
                # =================================================

                st.divider()

                st.header(
                    "📝 Réponse finale"
                )


                st.markdown(
                    final_answer
                )


                # =================================================
                # RÉSUMÉ
                # =================================================

                st.divider()

                st.subheader(
                    "📌 Résumé de l'exécution"
                )


                summary_col1, summary_col2 = (
                    st.columns(2)
                )


                with summary_col1:

                    st.write(
                        f"**Temps total :** "
                        f"{latency:.2f} s"
                    )

                    st.write(
                        f"**Modèle :** "
                        f"{llm_model}"
                    )

                    st.write(
                        f"**Appels LLM :** "
                        f"{llm_calls}"
                    )

                    st.write(
                        f"**Retries :** "
                        f"{retry_count}"
                    )


                with summary_col2:

                    st.write(
                        f"**Input tokens :** "
                        f"{input_tokens}"
                    )

                    st.write(
                        f"**Output tokens :** "
                        f"{output_tokens}"
                    )

                    st.write(
                        f"**Total tokens :** "
                        f"{total_tokens}"
                    )

                    st.write(
                        f"**Nombre d'outils :** "
                        f"{len(used_tools)}"
                    )


            # =====================================================
            # GESTION DES ERREURS
            # =====================================================

            except Exception as e:

                st.error(
                    "❌ Une erreur est survenue "
                    "pendant l'exécution de l'agent."
                )

                st.exception(e)


# ============================================================
# HISTORIQUE AGENTOPS
# ============================================================

st.divider()

st.header(
    "📈 Historique AgentOps"
)


history = load_run_history()


if history:

    # ========================================================
    # DERNIÈRES EXÉCUTIONS
    # ========================================================

    st.subheader(
        "📋 Historique des runs"
    )


    history_display = list(
        reversed(history)
    )


    for run in history_display:

        run_id = run.get(
            "run_id",
            "?"
        )

        duration = run.get(
            "duration",
            0
        )

        llm_calls_history = run.get(
            "llm_calls",
            0
        )

        model_history = run.get(
            "model",
            "qwen2.5:3b"
        )

        input_history = run.get(
            "input_tokens",
            0
        )

        output_history = run.get(
            "output_tokens",
            0
        )

        total_history = run.get(
            "total_tokens",
            0
        )

        tools_history = run.get(
            "tools",
            0
        )

        retries_history = run.get(
            "retries",
            0
        )

        errors_history = run.get(
            "errors",
            0
        )


        with st.expander(
            f"Run #{run_id} — {duration:.2f} s"
        ):

            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "⏱️ Temps",
                    f"{duration:.2f} s"
                )


            with col2:

                st.metric(
                    "🤖 LLM calls",
                    llm_calls_history
                )


            with col3:

                st.metric(
                    "🎟️ Tokens",
                    total_history
                )


            with col4:

                st.metric(
                    "🔄 Retries",
                    retries_history
                )


            st.write(
                f"**Modèle :** {model_history}"
            )

            st.write(
                f"**Input tokens :** "
                f"{input_history}"
            )

            st.write(
                f"**Output tokens :** "
                f"{output_history}"
            )

            st.write(
                f"**Nombre d'outils :** "
                f"{tools_history}"
            )


            if errors_history == 0:

                st.success(
                    "✅ Aucune erreur"
                )

            else:

                st.error(
                    f"❌ {errors_history} erreur(s)"
                )


    # ========================================================
    # COMPARAISON RAPIDE
    # ========================================================

    st.subheader(
        "📊 Comparaison des runs"
    )


    if len(history) >= 2:

        latest = history[-1]

        previous = history[-2]


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Temps dernier run",
                f"{latest.get('duration', 0):.2f} s",

                f"{latest.get('duration', 0) - previous.get('duration', 0):+.2f} s"
            )


        with col2:

            st.metric(
                "Retries",

                latest.get(
                    "retries",
                    0
                ),

                latest.get(
                    "retries",
                    0
                )
                -
                previous.get(
                    "retries",
                    0
                )
            )


        with col3:

            st.metric(
                "Tokens",

                latest.get(
                    "total_tokens",
                    0
                ),

                latest.get(
                    "total_tokens",
                    0
                )
                -
                previous.get(
                    "total_tokens",
                    0
                )
            )


        with col4:

            st.metric(
                "Outils",

                latest.get(
                    "tools",
                    0
                ),

                latest.get(
                    "tools",
                    0
                )
                -
                previous.get(
                    "tools",
                    0
                )
            )


    else:

        st.info(
            "Il faut au moins 2 runs pour "
            "comparer les performances."
        )


else:

    st.info(
        "Aucun historique disponible. "
        "Lancez votre première analyse."
    )