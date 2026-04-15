# ==============================
# AI Resume Screening System
# ==============================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.3
)

# Prompt Templates
extraction_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
Extract the following from the resume:
- Skills
- Tools
- Experience

Return in JSON format.

Resume:
{resume}
"""
)

matching_prompt = PromptTemplate(
    input_variables=["job_description", "candidate_data"],
    template="""
Compare candidate with job description.

Job Description:
{job_description}

Candidate:
{candidate_data}

Return:
- Matching percentage
- Missing skills

Do NOT assume skills not present.
"""
)

scoring_prompt = PromptTemplate(
    input_variables=["match_result"],
    template="""
Based on the match result below, assign a score between 0 to 100.

Match Result:
{match_result}

Only return number.
"""
)

explanation_prompt = PromptTemplate(
    input_variables=["resume", "job_description", "score"],
    template="""
Explain why this candidate received score {score}.

Resume:
{resume}

Job Description:
{job_description}

Do NOT assume missing skills.
"""
)

# Chains (LCEL)
parser = StrOutputParser()

extraction_chain = extraction_prompt | llm | parser
matching_chain = matching_prompt | llm | parser
scoring_chain = scoring_prompt | llm | parser
explanation_chain = explanation_prompt | llm | parser

# Pipeline
def run_pipeline(resume, job_description):

    print("\n--- STEP 1: Extraction ---")
    extracted = extraction_chain.invoke({"resume": resume})
    print(extracted)

    print("\n--- STEP 2: Matching ---")
    match = matching_chain.invoke({
        "job_description": job_description,
        "candidate_data": extracted
    })
    print(match)

    print("\n--- STEP 3: Scoring ---")
    score = scoring_chain.invoke({"match_result": match})
    print(score)

    print("\n--- STEP 4: Explanation ---")
    explanation = explanation_chain.invoke({
        "resume": resume,
        "job_description": job_description,
        "score": score
    })
    print(explanation)


# Data
job_description = """
Looking for Data Scientist with:
Python, Machine Learning, Deep Learning, SQL, Statistics
"""

resumes = {
    "Strong Candidate": "Python, Machine Learning, Deep Learning, SQL, Statistics, 3 years experience",
    "Average Candidate": "Python, Machine Learning, SQL",
    "Weak Candidate": "Excel, communication skills"
}

# Run
for name, resume in resumes.items():
    print(f"\n\n===== {name} =====")
    run_pipeline(resume, job_description)