from core.loader import (
    load_problems,
    get_problem_by_id,
    get_problems_by_category,
    get_problems_by_category_and_difficulty,
)
from core.wrong_book import (
    add_wrong_question,
    get_wrong_questions,
    get_wrong_stats,
)
from core.vector_store import build_vector_store
from core.rag import ask_rag


def get_all_problems():
    return load_problems()


def get_problem_detail(problem_id: int):
    problems = load_problems()
    problem = get_problem_by_id(problems, problem_id)

    if not problem:
        return None

    return {
        "id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "categories": problem.get("categories", []),
        "description": problem.get("description", ""),
        "idea": problem.get("idea", ""),
        "code": problem.get("code", ""),
    }


def recommend_problems(category: str, difficulty: str = None, num: int = 3):
    problems = load_problems()

    if difficulty:
        result = get_problems_by_category_and_difficulty(
            problems, category, difficulty
        )
    else:
        result = get_problems_by_category(problems, category)

    if not result:
        return []

    output = []
    for problem in result[:num]:
        output.append({
            "id": problem["id"],
            "title": problem["title"],
            "difficulty": problem["difficulty"],
            "categories": problem.get("categories", []),
        })

    return output


def add_problem_to_wrong_book(problem_id: int):
    problems = load_problems()
    problem = get_problem_by_id(problems, problem_id)

    if not problem:
        return None

    item = add_wrong_question(problem)

    return {
        "id": item["id"],
        "title": item["title"],
        "difficulty": item["difficulty"],
        "categories": item.get("categories", []),
        "wrong_count": item["wrong_count"],
    }


def get_wrong_book_list():
    wrong_questions = get_wrong_questions()

    if not wrong_questions:
        return []

    output = []
    for item in wrong_questions:
        output.append({
            "id": item["id"],
            "title": item["title"],
            "difficulty": item["difficulty"],
            "categories": item.get("categories", []),
            "wrong_count": item["wrong_count"],
        })

    return output


def get_wrong_book_stats():
    stats = get_wrong_stats()

    return {
        "total_wrong": stats.get("total_wrong", 0),
        "category_count": stats.get("category_count", []),
    }


def build_faiss_index():
    build_vector_store()
    return {"message": "FAISS 索引构建完成。"}


def ask_question(query: str, top_k: int = 3, model_name: str = "llama3"):
    answer = ask_rag(query=query, top_k=top_k, model_name=model_name)
    return {
        "query": query,
        "top_k": top_k,
        "model_name": model_name,
        "answer": answer,
    }