
from langchain_core.tools import tool
from contextvars import ContextVar


# ============================================================
# CONTEXTE DU CV DE L'UTILISATEUR
# ============================================================

# Chaque exécution possède son propre CV.
# On évite ainsi d'utiliser un CV global pour tous les utilisateurs.

CURRENT_CV = ContextVar(
    "current_cv",
    default=None
)


def set_current_cv(cv_text: str):
    """
    Définit le CV de l'utilisateur actuel.
    """

    CURRENT_CV.set(cv_text)


def get_current_cv():
    """
    Récupère le CV de l'utilisateur actuel.

    Si aucun CV n'a été fourni, on utilise temporairement
    data/cv.txt pour conserver la compatibilité avec
    l'ancien fonctionnement.
    """

    cv = CURRENT_CV.get()

    if cv is not None:
        return cv

    # Compatibilité avec l'ancien système
    with open(
        "data/cv.txt",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# ============================================================
# FONCTION INTERNE : CALCUL DE LA CORRESPONDANCE
# ============================================================

def calculate_skill_match():
    """
    Calcule :

    - les compétences présentes dans le CV
    - les compétences demandées par l'offre
    - les compétences correspondantes
    - les compétences manquantes
    - le taux de correspondance
    """

    # ========================================================
    # 1. RÉCUPÉRER LE CV DE L'UTILISATEUR
    # ========================================================

    cv = get_current_cv()


    # ========================================================
    # 2. LIRE L'OFFRE
    # ========================================================

    with open(
        "data/job_offer.txt",
        "r",
        encoding="utf-8"
    ) as f:

        offer = f.read()


    # ========================================================
    # 3. LISTE DES COMPÉTENCES
    # ========================================================

    skills = [

        "Python",

        "SQL",

        "Machine Learning",

        "Deep Learning",

        "Pandas",

        "NumPy",

        "Scikit-learn",

        "TensorFlow",

        "PyTorch",

        "Docker",

        "AWS",

        "Power BI",

        "Git"
    ]


    # ========================================================
    # 4. COMPÉTENCES DU CV
    # ========================================================

    cv_skills = {
        skill
        for skill in skills
        if skill.lower() in cv.lower()
    }


    # ========================================================
    # 5. COMPÉTENCES DE L'OFFRE
    # ========================================================

    job_skills = {
        skill
        for skill in skills
        if skill.lower() in offer.lower()
    }


    # ========================================================
    # 6. COMPÉTENCES CORRESPONDANTES
    # ========================================================

    matching = cv_skills.intersection(
        job_skills
    )


    # ========================================================
    # 7. COMPÉTENCES MANQUANTES
    # ========================================================

    missing = job_skills.difference(
        cv_skills
    )


    # ========================================================
    # 8. TAUX DE CORRESPONDANCE
    # ========================================================

    if len(job_skills) > 0:

        match_rate = (
            len(matching)
            / len(job_skills)
            * 100
        )

    else:

        match_rate = 0


    # ========================================================
    # 9. RETOUR
    # ========================================================

    return (
        matching,
        missing,
        match_rate
    )


# ============================================================
# TOOL 1 : RÉCUPÉRER LE CV
# ============================================================

@tool
def get_cv_skills() -> str:
    """
    Récupère le contenu du CV de l'utilisateur.
    """

    return get_current_cv()


# ============================================================
# TOOL 2 : RÉCUPÉRER L'OFFRE
# ============================================================

@tool
def get_job_offer() -> str:
    """
    Récupère le contenu de l'offre d'emploi.
    """

    with open(
        "data/job_offer.txt",
        "r",
        encoding="utf-8"
    ) as f:

        offer = f.read()

    return offer


# ============================================================
# TOOL 3 : COMPARER LES COMPÉTENCES
# ============================================================

@tool
def compare_skills() -> str:
    """
    Compare automatiquement le CV de l'utilisateur
    avec l'offre.

    Retourne :
    - les compétences correspondantes
    - les compétences manquantes

    Ne retourne PAS le score.
    """

    matching, missing, _ = (
        calculate_skill_match()
    )

    return f"""
Compétences correspondantes :
{sorted(matching)}

Compétences manquantes :
{sorted(missing)}
"""


# ============================================================
# TOOL 4 : ÉVALUER LA CANDIDATURE
# ============================================================

@tool
def evaluate_application() -> str:
    """
    Évalue globalement la candidature.

    Retourne :
    - le taux de correspondance
    - l'évaluation du profil
    """

    matching, missing, match_rate = (
        calculate_skill_match()
    )


    # ========================================================
    # ÉVALUATION
    # ========================================================

    if match_rate >= 80:

        evaluation = (
            "Très bon profil pour cette offre."
        )

    elif match_rate >= 60:

        evaluation = (
            "Profil intéressant, mais certaines "
            "compétences sont à renforcer."
        )

    else:

        evaluation = (
            "Le profil nécessite plusieurs "
            "améliorations avant de candidater."
        )


    return f"""
Taux de correspondance :
{match_rate:.0f}%

Évaluation :
{evaluation}
"""


# ============================================================
# TOOL 5 : CONSEILS D'AMÉLIORATION
# ============================================================

@tool
def get_improvement_advice() -> str:
    """
    Analyse les compétences manquantes
    et propose des axes d'amélioration.
    """

    _, missing, _ = (
        calculate_skill_match()
    )


    # ========================================================
    # SI AUCUNE COMPÉTENCE MANQUANTE
    # ========================================================

    if not missing:

        return """
Aucune compétence manquante détectée
par rapport aux compétences analysées.

Le profil correspond aux compétences
identifiées dans l'offre.
"""


    # ========================================================
    # CONSEILS
    # ========================================================

    advice = []


    for skill in sorted(missing):

        if skill == "Git":

            advice.append(
                "Apprendre Git et pratiquer le "
                "versionnement de projets avec GitHub."
            )

        elif skill == "Power BI":

            advice.append(
                "Développer les compétences Power BI "
                "à travers des projets d'analyse "
                "et de visualisation de données."
            )

        elif skill == "Python":

            advice.append(
                "Renforcer Python à travers des "
                "projets pratiques en Data Science."
            )

        elif skill == "SQL":

            advice.append(
                "Renforcer SQL avec des exercices "
                "de requêtes et de manipulation de données."
            )

        elif skill == "Machine Learning":

            advice.append(
                "Approfondir le Machine Learning "
                "avec des projets pratiques."
            )

        elif skill == "Deep Learning":

            advice.append(
                "Approfondir le Deep Learning "
                "avec des projets CNN, RNN ou Transformers."
            )

        elif skill == "Docker":

            advice.append(
                "Apprendre Docker et pratiquer "
                "la conteneurisation de projets."
            )

        elif skill == "AWS":

            advice.append(
                "Développer les connaissances AWS "
                "à travers des projets Cloud."
            )

        elif skill == "Pandas":

            advice.append(
                "Pratiquer Pandas sur des datasets "
                "réels de Data Science."
            )

        elif skill == "NumPy":

            advice.append(
                "Renforcer NumPy avec des exercices "
                "de manipulation de tableaux."
            )

        elif skill == "Scikit-learn":

            advice.append(
                "Pratiquer Scikit-learn sur des projets "
                "de Machine Learning."
            )

        elif skill == "TensorFlow":

            advice.append(
                "Pratiquer TensorFlow sur des projets "
                "de Deep Learning."
            )

        elif skill == "PyTorch":

            advice.append(
                "Pratiquer PyTorch sur des projets "
                "de Deep Learning."
            )

        else:

            advice.append(
                f"Développer la compétence {skill} "
                "à travers un projet pratique."
            )


    # ========================================================
    # CONSTRUIRE LA RÉPONSE
    # ========================================================

    result = "\n".join(
        f"- {item}"
        for item in advice
    )


    return f"""
Compétences à améliorer :
{sorted(missing)}

Conseils :
{result}
"""
