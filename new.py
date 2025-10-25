
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
desc = """
What is Data in AI? (Foundation)

Define data and explain its role in AI decision-making.
Explain why statistics and probability are essential in AI.
Calculate mean, median, and mode from a dataset.
Interpret which measure best represents the dataset.
Build awareness that different measures can tell different “stories.”

PPT Activity: “Survey to Summary”

Slide 1: Fictional dataset (10 students’ marks).

Slide 2: Ask learner to compute mean, median, mode step by step (with hints).

Slide 3: Show the same data as bar chart, pie chart, and line chart.

Task: Learner selects which chart best tells “Which subject is most popular?” and explains why."""

def generate_positive_reply():

    prompt = f"""
    Given the following template. I want to create a similar one for a new description.
    First look at the example.
    
    [Tutor Actions:
    Show a realistic mini-dataset (e.g., House Data: Size, Location, Number of Rooms → Price).
    Ask: “If I wanted to guess the price of a new house, which column do you think I’d need as the answer? Which ones would help me get there?”
    Encourage students to label the dataset informally with sticky notes or by circling answers on the board.
    Student Actions:
    Observe the dataset and attempt to identify the “answer column” (target) and the “clue columns” (features).
    Share reasoning with peers before answering aloud.
    Tutor Prompt:
    “If the target is what we want to predict, then what role do the other columns play?”
    “Can the model learn only from the answer column, or does it need the clues too?”
    Possible Confusion:
    Students may think the dataset only needs the answer (target) to work.
    They may confuse “label” with “feature.”
    Tutor Guidance:
    Clarify: “The model needs examples of both clues (features) and the correct answer (labels) during training. Later, it will try to predict the target when only features are given.”

    Use analogy: Features are like ingredients, labels are the recipe’s final dish, and the target is what you want to make again.]


    The above example is being used in a lesson doc.
    I will share a description of the lesson please generate a similar one for me, considering the lesson description shared.
    The abve example is to showcase that I need, Tutor Actions, Student Actions, Tutor Prompt, Possible Confusion.
    And the sample data under these heading represent just an example of what kind of data I am expecting.

    Make sure the one you generate is professionaly written, it is creative and alines with the descripton and the activity.

    Here is the description:
    {desc}

   """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

print(generate_positive_reply())