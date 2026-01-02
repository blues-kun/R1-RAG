"""
Prompt Templates for R1-RAG

Contains carefully designed prompts for:
1. Training prompt template (guides model output format)
2. GPT-4o annotation generation (creates golden plans)
"""

# One-shot example demonstrating the expected output format
ONE_SHOT_EXAMPLE = """## One-Shot Example:
Input:
Who was the screenwriter of the film directed by the person who created the Money in the Bank ladder match?

Output:
<think> The question involves multiple entities and relations, so it is best decomposed into smaller sub-questions. First, I need to identify the creator of the Money in the Bank ladder match. Then, I should check which film that person directed. Finally, I must find the screenwriter of that film. </think>
<plan>
{ "Q1": ["Who created the Money in the Bank ladder match?", "#1"], "Q2": ["Which film was directed by #1?", "#2"], "Q3": ["Who was the screenwriter of #2?", "#3"] } 
</plan>

<subPlan>
    <think> To start, I need to find who created the Money in the Bank ladder match. </think>
    <search> creator of the Money in the Bank ladder match </search>
    <information> The Money in the Bank ladder match was created by Chris Jericho. </information>
    <think> The information shows that Chris Jericho is the creator. </think>
    <subAnswer> #1 = Chris Jericho </subAnswer>
</subPlan>

<subPlan>
    <think> Next, I need to find which film Chris Jericho directed, based on the previous answer. </think>
    <search> Which film was directed by Chris Jericho </search>
    <information> Chris Jericho directed the film "But I'm Chris Jericho!". </information>
    <think> The evidence indicates that the film directed by Chris Jericho is "But I'm Chris Jericho!". </think>
    <subAnswer> #2 = "But I'm Chris Jericho!" </subAnswer>
</subPlan>

<subPlan>
    <think> Finally, I should determine who wrote the film "But I'm Chris Jericho!". </think>
    <search> "But I'm Chris Jericho!" film screenwriter </search>
    <information> The series "But I'm Chris Jericho!" was written by Bob Kerr and Norm Hiscock. </information>
    <think> The results confirm that the screenwriters of the film are Bob Kerr and Norm Hiscock. </think>
    <subAnswer> #3 = Bob Kerr and Norm Hiscock </subAnswer>
</subPlan>

<think> I have gathered all the necessary information from the sub-questions and can now provide the final answer. </think>
<answer> Bob Kerr and Norm Hiscock </answer>
"""

# Main training prompt template
PLANNING_PROMPT_TEMPLATE = """Answer the given question by following the steps below. You must conduct all reasoning inside <think> and </think> before producing <plan>, <search>, <subAnswer> and <answer>. 

Step 1: Explicitly generate one or more sub-questions within the <plan> and </plan> block.
 - Each sub-question must contain both a question and a placeholder (#1, #2, etc.) that represents the answer to that question.
 - Each sub-question should be as brief and precise as possible.
 - If a sub-question depends on the answer to a previous one, use a placeholder (#1, #2, etc.) to represent that dependency.
 - The output format of the sub-questions must follow this JSON structure:
{
    "Q1": ["First sub-question", "#1"],
    "Q2": ["Second sub-question using #1", "#2"],
    ...
}

Step 2: For each sub-question, create a block enclosed in <subPlan> and </subPlan>.
Within each <subPlan> block you must:
 - In sequential order, take one sub-question from <plan> and fill it between <search> and </search>.
 - If you lack some knowledge, call a search engine using <search> query </search>. The search engine will return results enclosed in <information> and </information>. You may search as many times as needed.
 - Conclude the block with a <subAnswer> that binds the answer to the current sub-question.

Step 3: Provide the final result inside <answer> and </answer>, without detailed explanations.

{one_shot_example}

## Now, it's your turn! Please answer the following question!!!
Question: {question}
"""

# GPT-4o prompt for generating golden plan annotations
GPT4O_PLAN_ANNOTATION_PROMPT = """You are an expert at decomposing complex multi-hop questions into structured reasoning plans.

Given a multi-hop question and its gold answer, generate:
1. A plan (DAG) that decomposes the question into sequential sub-questions
2. The intermediate answer for each sub-question

IMPORTANT RULES:
- Each sub-question should retrieve ONE piece of information
- Use placeholders (#1, #2, etc.) to reference previous answers
- The final sub-question's answer should match the gold answer
- Ensure the dependency structure is correct (later questions depend on earlier ones)

Output format (JSON):
{{
    "plan": {{
        "Q1": ["<sub-question 1>", "#1"],
        "Q2": ["<sub-question 2 using #1>", "#2"],
        ...
    }},
    "graph": {{
        "Q1": {{"answer": "<intermediate answer 1>"}},
        "Q2": {{"answer": "<intermediate answer 2>"}},
        ...
    }}
}}

Question: {question}
Gold Answer: {gold_answer}

Generate the plan and execution graph:
"""

# Prompt for validating generated plans
PLAN_VALIDATION_PROMPT = """Given the following question, plan, and gold answer, verify if the plan correctly leads to the answer.

Question: {question}
Gold Answer: {gold_answer}

Generated Plan:
{plan}

Execution Graph:
{graph}

Is this plan valid? (Answer YES or NO with brief explanation)
"""

