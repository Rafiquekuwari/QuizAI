from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import re
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Quiz Question Generator API",
    description="API to generate multiple-choice questions using T5-small",
)

# Enable CORS to allow React frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your Netlify URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class QuizRequest(BaseModel):
    topic: str
    num_questions: int

# Initialize T5 model
question_generator = pipeline("text2text-generation", model="t5-small")

# Function to clean and format generated output
def format_question(text: str) -> str:
    try:
        # Assuming output: "Question: ... A) ... B) ... C) ... D) ... Correct Answer: ..."
        question_match = re.match(
            r"(Question:.*?)(A\).*?)(B\).*?)(C\).*?)(D\).*?)(Correct Answer:.*)",
            text,
            re.DOTALL,
        )
        if question_match:
            parts = question_match.groups()
            return (
                f"{parts[0].strip()}\n{parts[1].strip()}\n{parts[2].strip()}\n"
                f"{parts[3].strip()}\n{parts[4].strip()}\n{parts[5].strip()}"
            )
        return text + "\n(Note: Output format may vary, please verify!)"
    except:
        return text + "\n(Note: Unable to parse output, please verify!)"

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Quiz Generator API is running"}

# Generate questions endpoint
@app.post("/generate-questions")
async def generate_questions(request: QuizRequest):
    if not request.topic or request.num_questions < 1 or request.num_questions > 5:
        raise HTTPException(
            status_code=400,
            detail="Topic is required and number of questions must be between 1 and 5",
        )

    try:
        questions = []
        for _ in range(request.num_questions):
            prompt = (
                f"generate question: Create a multiple-choice question about {request.topic} "
                "with 4 options and the correct answer."
            )
            output = question_generator(
                prompt, max_length=150, num_beams=5, temperature=0.7
            )[0]["generated_text"]
            formatted = format_question(output)
            questions.append(formatted)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")