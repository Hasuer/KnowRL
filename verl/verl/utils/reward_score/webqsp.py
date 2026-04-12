import re
from rapidfuzz import fuzz
from openai import OpenAI

client = OpenAI(
    base_url="https://api.lingleap.com/v1",
    api_key="sk-lnFlANzlUmGKx8GAfW7u1uZbinFaFiICBD4E1vM8qMypRsJx"
)

# format check
def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    Args:
        processed_str: Processed response string from the model
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True
    validation_begin = True

    # Check the begin <think> tag
    if not processed_str.lstrip().startswith("<think>"):
        validation_begin = False

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)
        
        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_passed = False

    return validation_passed, validation_begin

def normalize_answer(text: str) -> str:
    """统一格式：去标点、空格、小写化"""
    text = text.lower().strip()
    # 去掉所有标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    return text

# def em_check(predicted: str, target: str, tol: float = 1e-6) -> bool:
#     """兼容数值近似匹配 + 字符完全匹配"""
#     import pdb; pdb.set_trace()
#     if predicted is None or target is None:
#         return False
    
#     predicted = predicted.strip()
#     target = target.strip()

#     # 数值型匹配
#     try:
#         return abs(float(predicted) - float(target)) < tol
#     except ValueError:
#         pass  # 如果不是数值，就走文本匹配

#     # 文本型匹配
#     return normalize_answer(predicted) == normalize_answer(target)

def em_check(predicted: str, target: str, tol: float = 1e-6) -> bool:
    """兼容数值近似匹配 + 字符完全匹配 + 语义包含匹配"""
    if predicted is None or target is None:
        return False
    
    predicted = predicted.strip().lower()
    target = target.strip().lower()

    # 数值型匹配
    try:
        return abs(float(predicted) - float(target)) < tol
    except ValueError:
        pass  # 如果不是数值，就走文本匹配

    # 文本型匹配（包含匹配）
    p_norm = normalize_answer(predicted)
    t_norm = normalize_answer(target)

    if p_norm == t_norm:
        return True

    # 如果 predicted 是 target 的子串，或者 target 是 predicted 的子串
    if p_norm in t_norm or t_norm in p_norm:
        return True
    
    # 模糊匹配
    score = fuzz.partial_ratio(predicted, target)
    return score >= 85  # 比如设定85分为匹配阈值

def extract_solution(solution_str: str) -> str:
    """
    从模型输出中提取 <answer> 标签内的最终答案。
    如果未找到 <answer>，则尝试从最后一行提取数字或文字答案。
    """
    if not solution_str:
        return ""
    
    # 提取所有 <answer> ... </answer>
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", solution_str, re.DOTALL)
    if matches:
        # 返回最后一个匹配项
        return matches[-1].strip()
    
    # 如果没有显式标签，则取最后一行作为 fallback
    lines = [l.strip() for l in solution_str.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        # 处理类似 “The answer is 42.” 这种
        m = re.search(r"(-?\d+(\.\d+)?|\b[A-Za-z]+\b)$", last_line)
        if m:
            return m.group(1)
        return last_line
    return ""

# 加入 format reward
def compute_score_v2(solution_str, ground_truth, format_score=0.2, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    format_valid, being_valid = validate_response_structure(solution_str)

    if answer is None:  # 如果没有提取到答案
        return 0.0
    else:
        answer_correct = False
        for target in ground_truth["target"]:
            if em_check(answer, target):
                answer_correct = True
                break
        
    # 最终的 format_score 是根据 format_valid 和 being_valid 的加权，比重分别是 0.75 和 0.25
    format_score_final = (0.75 * format_valid + 0.25 * being_valid) * format_score
    if answer_correct:
        final_score = score - (format_score - format_score_final)
    else:
        final_score = format_score_final
    
    return final_score


def compute_score_genrm_api(solution_str, ground_truth, format_score=0.2, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)

    format_valid, being_valid = validate_response_structure(solution_str)
    
    response = client.responses.create(
        model="gpt-5",
        input="Hi"
    )
    print(response)

    if answer is None:  # 如果没有提取到答案
        return 0.0
    else:
        answer_correct = False
        for target in ground_truth["target"]:
            if em_check(answer, target):
                answer_correct = True
                break
        if not answer_correct:
            # genrm api调用
            response = client.responses.create(
                model="gpt-5",
                input="Hi"
            )

            print(response)

        
    # 最终的 format_score 是根据 format_valid 和 being_valid 的加权，比重分别是 0.75 和 0.25
    format_score_final = (0.75 * format_valid + 0.25 * being_valid) * format_score
    if answer_correct:
        final_score = score - (format_score - format_score_final)
    else:
        final_score = format_score_final
    
    return final_score
