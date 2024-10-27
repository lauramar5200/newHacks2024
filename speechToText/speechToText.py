# Import libraries for recording and saving audio
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import speech_recognition as sr
import spacy
from transformers import pipeline


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


# Define parameters for audio recording
freq = 44100  # Sample rate
duration = 20  # Duration of recording in seconds


# Record audio with sounddevice
print("Recording audio...")
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2, device=0)
sd.wait()  # Wait for the recording to finish
print("Recording finished!")


# Save the recording as a WAV file
write("recording0.wav", freq, recording)  # Saves with scipy
wv.write("recording1.wav", recording, freq, sampwidth=2)  # Saves with wavio


# Transcribe the recorded audio
filename = "recording1.wav"
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data)
        print("Transcription successful.")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        text = ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        text = ""


# Write the transcription to a text file
with open("transcription.txt", "w") as f:
    f.write(text)


from transformers import pipeline


# Initialize the question-answering pipeline with a pre-trained BERT model
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")


# Input text (replace this with your transcribed string)
context = """
Hi my name is Stacy. I recently bought a new laptop, but it crashes all the time when I tried to go over 100 megabytes.
I love the coffee shop down the street; it's pretty good, but I have a problem which only occurs when I have other applications open,
which could indicate a memory usage problem. I appreciate guidance on any troubleshooting steps for recommendations to resolve this issue.
Thank you in advance.
"""


# Reduced list of questions to extract information
questions = [
    "What is the name?",
    "What is the problem?",
    "Any approaches customer took to solve?",
    "Any additional notes?",
    "What is the mood of the customer?",
    "Suggested replies to customer"
]
questions_for_prompts = [
    "What is the name?", #Works well
    "What is the problem in a short sentence?",
    "How and where the person attempted to solve the problem?",
    "Any additional notes related to the problem without including anything unrelated problem and do not include any non-facts or pleasentaries and do not include exact copies of previous attempts",
    "What is the mood of the user based on their word choice and word connotations described in one word like happy or sad or impatient or etc and not saying their name?",
    "What are some possible ways to solve the user's problem described in a problem but the solution is not described in the problem and it should be newly created by the prompt"
]


# Map detailed prompts to general questions
prompt_to_general = dict(zip(questions_for_prompts, questions))


# Dictionary to store the answers
answers = {}


# Extract answers for each detailed question prompt
for question_prompt in questions_for_prompts:
    try:
        result = qa_pipeline(question=question_prompt, context=text)
        general_question = prompt_to_general[question_prompt]
        answers[general_question] = result['answer']
    except Exception as e:
        print(f"An error occurred while processing question: {question_prompt}. Error: {e}")
        general_question = prompt_to_general[question_prompt]
        answers[general_question] = "Error occurred"


# Print the structured output with general questions
print("Categorized Summary:")
for general_question, answer in answers.items():
    print(f"{general_question}: {answer}")
