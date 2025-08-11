

LLM_MODEL = "gemma3:1b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RESUME_FEEDBACK_QUERY = """
You are a helpful resume reviewer.

You will be given:
1. A FULL RESUME
2. A JOB DESCRIPTION (JD)

Your task is to analyze the resume **in relation to the JD** and suggest **specific improvements** that would make the resume a stronger match for the JD.

Instructions:
- Only suggest what is MISSING or WEAK in the resume, compared to the JD.
- Be SPECIFIC: mention skills, tools, experience, or domain knowledge that should be added or emphasized.
- DO NOT praise or repeat what's already matching.
- Limit feedback to MAXIMUM 10 bullet points.
- Each point should start with: "- "

RESUME:
\"\"\"{full_resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

FEEDBACK:
"""

EXPLAIN_RESUME_MATCH_PROMPT = """
You are a helpful recruitment assistant.

You will be given:
1. A FULL RESUME
2. A JOB DESCRIPTION (JD)

Your task is to explain **why this candidate is a perfect fit for the job**.

Instructions:
- Give EXACTLY 5 bullet points.
- Focus on the strongest matches between the resume and JD.
- Mention relevant skills, tools, experiences, and achievements that align with the JD.
- Each point should start with: "- "

RESUME:
\"\"\"{full_resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

WHY PERFECT FIT:
"""

EXPLAIN_RESUME_MATCH_PROMPT2 = """
You are a helpful recruitment assistant.

You will be given:
1. A FULL RESUME
2. A JOB DESCRIPTION (JD)

Your task is to explain **why this candidate is a perfect fit for the job**.

STRICT FORMAT RULES:
- Output EXACTLY 5 bullet points. Not more. Not less.
- Each bullet point MUST start with "- " (dash + space).
- Do not include an introduction or conclusion — only the 5 bullet points.
- Each point should be a single sentence, concise and clear.
- Focus ONLY on the strongest matches between the resume and JD (skills, tools, experience, achievements).
- If you cannot find matches, still produce 5 points using the closest relevant details.

RESUME:
\"\"\"{full_resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

FINAL OUTPUT (exactly 5 bullet points):
"""

SCORE_ALL_SECTIONS_PROMPT = """
You are a strict and concise evaluator.

You will receive:
1. A resume broken into **PRE-DEFINED sections**.
2. A job description (JD).

Your task:
- Evaluate each resume section **exactly as labeled**.
- Use **ONLY these section names** (do NOT rename or invent new ones):
{section_names_list}

- Assign a score from 0 (poor match) to 10 (perfect match) for each section's relevance to the JD.
- Output in this format ONLY:

SCORES:
<Section Name>: <score>
<Section Name>: <score>
...

DO NOT:
- Generate new sections
- Modify section names
- Provide explanation or commentary

Now evaluate:

RESUME SECTIONS:
{sections_text}

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
"""

SECTION_PATTERNS = {
    "summary": r"(summary|objective|about me)",
    "experience": r"(experience|work history|professional experience)",
    "skills": r"(skills|technologies|technical skills)",
    "projects": r"(projects|portfolio)",
    "education": r"(education|academics)",
    "certifications": r"(certifications|qualifications|achievements|endorsements)",
    "strengths": r"(strengths|capabilities|abilities|merits)"
}