from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
from pathlib import Path


from services.assistant_service import (
    ask_question,
    recommend_problems,
    get_problem_detail,
    add_problem_to_wrong_book,
    get_wrong_book_list,
    get_wrong_book_stats,
    build_faiss_index,
)

app = FastAPI(title="Study Assistant API")
templates = Jinja2Templates(directory="web/templates")

def load_page_options():
    problems_path = Path("data/processed/problems.json")
    categories_path = Path("data/processed/categories.json")

    problems = []
    categories = []

    print("problems_path:", problems_path.resolve(), problems_path.exists())
    print("categories_path:", categories_path.resolve(), categories_path.exists())

    if problems_path.exists():
        with open(problems_path, "r", encoding="utf-8") as f:
            problems = json.load(f)

    if categories_path.exists():
        with open(categories_path, "r", encoding="utf-8") as f:
            category_data = json.load(f)
            categories = list(category_data.keys())

    problems = sorted(problems, key=lambda x: x.get("id", 0))
    categories = sorted(categories)

    print("problems count:", len(problems))
    print("categories count:", len(categories))

    return problems, categories


def build_page_context(request: Request, **kwargs):
    problems, categories = load_page_options()

    context = {
        "request": request,
        "problems": problems,
        "categories": categories,

        "ask_result": None,
        "ask_query": "",

        "problem_result": None,
        "problem_id": "",

        "recommend_results": None,
        "recommend_category": "",
        "recommend_difficulty": "",
        "recommend_num": 3,

        "add_wrong_result": None,
        "wrong_id": "",

        "wrong_list": get_wrong_book_list(),
        "wrong_stats": get_wrong_book_stats(),

        "build_index_result": None,
    }

    context.update(kwargs)
    return context

class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    model_name: str = "llama3"


class WrongRequest(BaseModel):
    problem_id: int


@app.get("/")
def root():
    return RedirectResponse(url="/page")


# ----------------------------
# 网页首页
# ----------------------------
@app.get("/page", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        build_page_context(request)
    )

# ----------------------------
# 网页：RAG问答
# ----------------------------
@app.post("/page/ask", response_class=HTMLResponse)
def page_ask(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(3),
    model_name: str = Form("llama3"),
):
    result = ask_question(query=query, top_k=top_k, model_name=model_name)

    return templates.TemplateResponse(
        "index.html",
        build_page_context(
            request,
            ask_result=result["answer"],
            ask_query=query,
        )
    )


# ----------------------------
# 网页：题目详情查询
# ----------------------------

@app.post("/page/problem", response_class=HTMLResponse)
def page_problem(
    request: Request,
    problem_id: int = Form(...),
):
    result = get_problem_detail(problem_id)

    return templates.TemplateResponse(
        "index.html",
        build_page_context(
            request,
            problem_result=result,
            problem_id=problem_id,
        )
    )

# ----------------------------
# 网页：分类推荐
# ----------------------------
@app.post("/page/recommend", response_class=HTMLResponse)
def page_recommend(
    request: Request,
    category: str = Form(...),
    difficulty: str = Form(""),
    num: int = Form(3),
):
    difficulty_value = difficulty.strip() or None
    results = recommend_problems(category, difficulty_value, num)

    return templates.TemplateResponse(
        "index.html",
        build_page_context(
            request,
            recommend_results=results,
            recommend_category=category,
            recommend_difficulty=difficulty,
            recommend_num=num,
        )
    )

# ----------------------------
# 网页：添加错题
# ----------------------------
@app.post("/page/wrong/add", response_class=HTMLResponse)
def page_add_wrong(
    request: Request,
    problem_id: int = Form(...),
):
    result = add_problem_to_wrong_book(problem_id)

    return templates.TemplateResponse(
        "index.html",
        build_page_context(
            request,
            add_wrong_result=result,
            wrong_id=problem_id,
        )
    )

# ----------------------------

# ----------------------------
@app.get("/problem/{problem_id}")
def get_problem(problem_id: int):
    result = get_problem_detail(problem_id)
    if result is None:
        raise HTTPException(status_code=404, detail="未找到该题目")
    return result


@app.get("/recommend")
def recommend(category: str, difficulty: str = None, num: int = 3):
    results = recommend_problems(category, difficulty, num)
    return {"results": results}


@app.post("/ask")
def ask(req: AskRequest):
    return ask_question(req.query, req.top_k, req.model_name)


@app.post("/wrong/add")
def add_wrong(req: WrongRequest):
    result = add_problem_to_wrong_book(req.problem_id)
    if result is None:
        raise HTTPException(status_code=404, detail="未找到该题目")
    return result


@app.get("/wrong/list")
def wrong_list():
    return {"results": get_wrong_book_list()}


@app.get("/wrong/stats")
def wrong_stats():
    return get_wrong_book_stats()


@app.post("/build_index")
def build_index():
    return build_faiss_index()

