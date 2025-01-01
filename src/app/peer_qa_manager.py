import os
import json
import uuid

PEER_QA_FILE = "src/app/peer_qa.json"


def load_peer_qa_data() -> dict:
    """Load peer questions from JSON. Return dict with 'peer_questions' list if none found."""
    if os.path.exists(PEER_QA_FILE):
        try:
            with open(PEER_QA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "peer_questions" not in data:
                    data["peer_questions"] = []
                for q in data["peer_questions"]:
                    q.setdefault("voters", [])
                    for ans in q.get("answers", []):
                        ans.setdefault("voters", [])
                return data
        except Exception:
            return {"peer questions": []}
    else:
        return {"peer_questions": []}


def save_peer_qa_data(data: dict):
    """Save peer questions to JSON"""
    os.makedirs(os.path.dirname(PEER_QA_FILE), exist_ok=True)
    with open(PEER_QA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def post_question(question_text: str, skill_tag: str) -> bool:
    """
    Post a new question. Returns True if posted successfully.
    We only do 'anonymous' posting: no user info is stored except for upvote checks.
    """
    if not question_text.strip():
        return False

    data = load_peer_qa_data()

    new_question = {
        "id": str(uuid.uuid4()),
        "question_text": question_text.strip(),
        "skill_tag": skill_tag,
        "upvotes": 0,
        "answers": [],
        "voters": []
    }

    data["peer_questions"].append(new_question)
    save_peer_qa_data(data)
    return True


def post_answer(question_id: str, answer_text: str, user_uak: str) -> bool:
    """Post a new answer to an existing question. Returns True if success."""
    if not answer_text.strip():
        return False

    data = load_peer_qa_data()
    for q in data["peer_questions"]:
        if q["id"] == question_id:
            new_answer = {
                "answer_text": answer_text.strip(),
                "upvotes": 0,
                "voters": []
            }
            q["answers"].append(new_answer)
            save_peer_qa_data(data)
            return True
    return False


def upvote_question(question_id: str, user_uak: str) -> bool:
    """
    Increment upvote count for a question. Returns True if success.
    Prevent multiple upvotes by the same UAK.
    """
    if not user_uak:
        return False

    data = load_peer_qa_data()
    for q in data["peer_questions"]:
        if q['id'] == question_id:
            q.setdefault("voters", [])
            if user_uak in q.get("voters", []):
                return False

            q["upvotes"] = q.get("upvotes", 0) + 1
            q["voters"].append(user_uak)
            save_peer_qa_data(data)
            return True

    return False


def upvote_answer(question_id: str, answer_text: str, user_uak: str) -> bool:
    """
    Increment upvote for a specific answer in a question
    Prevent multiple upvotes by the same UAK.
    """
    if not user_uak:
        return False

    data = load_peer_qa_data()
    for q in data["peer_questions"]:
        if q["id"] == question_id:
            for ans in q["answers"]:
                if ans["answer_text"] == answer_text:
                    ans.setdefault("voters", [])
                    if user_uak in ans.get("voters", []):
                        return False

                    ans["upvotes"] = ans.get("upvotes", 0) + 1
                    ans.setdefault("voters", []).append(user_uak)
                    save_peer_qa_data(data)
                    return True

    return False


def search_questions(search_term: str, skill_filter: str = "All") -> list:
    """
    Return a filtered list of questions.
      - If skill_filter != "All", only questions with matching skill_tag
      - If search_term is not empty, only those whose text contains the term
    """
    data = load_peer_qa_data()
    all_qs = data["peer_questions"]
    filtered = []
    search_term = search_term.strip().lower()

    for q in all_qs:
        if skill_filter != "All" and q["skill_tag"] != skill_filter:
            continue

        if search_term:
            if search_term not in q["question_text"].lower():
                continue

        filtered.append(q)

    return filtered
