from tools.cv_tools import (
    set_current_cv,
    get_cv_skills,
    get_job_offer,
    compare_skills,
    evaluate_application,
    get_improvement_advice
)


def test_get_cv_skills():
    cv = """
    Compétences :
    Python
    SQL
    Machine Learning
    """

    set_current_cv(cv)

    result = get_cv_skills.invoke({})

    assert "Python" in result
    assert "SQL" in result
    assert "Machine Learning" in result


def test_get_job_offer():
    result = get_job_offer.invoke({})

    assert result is not None
    assert len(result) > 0


def test_compare_skills():
    cv = """
    Compétences :
    Python
    SQL
    Machine Learning
    """

    set_current_cv(cv)

    result = compare_skills.invoke({})

    assert "Compétences correspondantes" in result
    assert "Compétences manquantes" in result


def test_evaluate_application():
    cv = """
    Compétences :
    Python
    SQL
    Machine Learning
    """

    set_current_cv(cv)

    result = evaluate_application.invoke({})

    assert "Taux de correspondance" in result
    assert "Évaluation" in result


def test_get_improvement_advice():
    cv = """
    Compétences :
    Python
    SQL
    Machine Learning
    """

    set_current_cv(cv)

    result = get_improvement_advice.invoke({})

    assert result is not None
    assert len(result) > 0