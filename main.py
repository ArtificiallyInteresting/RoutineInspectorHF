import streamlit as st
import pandas as pd
import numpy as np
import langchain as lc
import examples
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

st.title('Routine Inspector')

st.header("What's your current routine?")

st.text("Please enter your routine below using the format provided:")
routine = st.text_area("Routine", """A:
Barbell Back Squats
Barbell Front Squats
Hip Thrust
Plank
Crunches

B:
Deadlift
Barbell Stiff-Leg Deadlift
Pull-Ups
Face Pulls
Dumbbell curls
Hammer Curls

C:
Standing Barbell Military Press
Seated Barbell Shoulder Press
Flat Dumbbell Chest Flys
Bench Press""")

@st.cache_data
def load_prompt():
    prompt = open("prompt.txt", "r")
    return prompt.read()

prompt_text = load_prompt()

template_text = "Routine: \n {routine}\n Response: \n {response}"

example_prompt = lc.PromptTemplate(input_variables=["routine", "response"], template=template_text)

few_shot_examples = examples.get_examples()


# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     # This is the list of examples available to select from.
#     few_shot_examples,
#     # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
#     OpenAIEmbeddings(),
#     # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
#     Chroma,
#     # This is the number of examples to produce.
#     k=1
# )

# st.write(example_prompt.format(**few_shot_examples[0]))
prompt = lc.FewShotPromptTemplate(
    # example_selector=example_selector, 
    examples=few_shot_examples, 
    example_prompt=example_prompt, 
    suffix="Routine: \n {routine}\n Response: \n", 
    input_variables=["routine"],
)

st.write(prompt.format(routine=routine))

