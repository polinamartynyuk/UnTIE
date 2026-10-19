"""Unicode-aware metrics missing from the historical English evaluator."""
import re


def word_tokens(text):
    return re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)


def rouge_l_f1(prediction, reference):
    left, right = word_tokens(prediction), word_tokens(reference)
    if not left or not right:
        return 1.0 if left == right else 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    return 2.0 * lcs / (len(left) + len(right))


def apply_unicode_rouge(row, gold_answers):
    answer = str(row.get("qa_answer") or "")
    refs = [str(x) for x in gold_answers]
    row["rouge_l_f1"] = max((rouge_l_f1(answer, ref) for ref in refs), default=0.0)
    row["rouge_l_impl"] = "unicode_word_lcs_v1"
    return row
