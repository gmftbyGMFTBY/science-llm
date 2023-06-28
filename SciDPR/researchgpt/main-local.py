import sys
import ipdb
sys.path.append('../scidpr')
from model import Agent, SciDPR 

from flask import Flask, request, render_template
from io import BytesIO
from PyPDF2 import PdfReader
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class Chatbot():

    def __init__(self):
        args = {
            'question_model': '/home/johnlan/SciDPR/ckpt/qasper/scidpr/question_encoder',
            'answer_model': '/home/johnlan/SciDPR/ckpt/qasper/scidpr/answer_encoder',
            'q_max_len': 128,
            'a_max_len': 256,
            'mode': 'test'
        }
        model = SciDPR(**args)
        self.agent = Agent(model, args) 
        print(f'[!] ========== load the model and agent over ==========')
    
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        for i in range(number_of_pages):
            page = pdf.pages[i]
            page_text = []

            def visitor_body(text, cm, tm, fontDict, fontSize):
                x = tm[4]
                y = tm[5]
                # ignore header/footer
                if (y > 50 and y < 720) and (len(text.strip()) > 1):
                    page_text.append({
                    'fontsize': fontSize,
                    'text': text.strip().replace('\x03', ''),
                    'x': x,
                    'y': y
                    })

            _ = page.extract_text(visitor_text=visitor_body)

            blob_font_size = None
            blob_text = ''
            processed_text = []

            for t in page_text:
                if t['fontsize'] == blob_font_size:
                    blob_text += f" {t['text']}"
                    if len(blob_text) >= 2000:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                        blob_font_size = None
                        blob_text = ''
                else:
                    if blob_font_size is not None and len(blob_text) >= 1:
                        processed_text.append({
                            'fontsize': blob_font_size,
                            'text': blob_text,
                            'page': i
                        })
                    blob_font_size = t['fontsize']
                    blob_text = t['text']
                paper_text += processed_text
        print("Done parsing paper")
        # print(paper_text)
        return paper_text

    def paper_df(self, pdf):
        print('Creating dataframe')
        filtered_pdf= []
        for row in pdf:
            if len(row['text']) < 30:
                continue
            filtered_pdf.append(row)
        df = pd.DataFrame(filtered_pdf)
        # print(df.shape)
        # remove elements with identical df[text] and df[page] values
        df = df.drop_duplicates(subset=['text', 'page'], keep='first')
        df['length'] = df['text'].apply(lambda x: len(x))
        print('Done creating dataframe')
        return df

    def calculate_embeddings(self, df):
        print('Calculating embeddings')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        texts = df.text.values.tolist()
        embeddings = self.agent.get_embedding(texts, question=False, inner_bsz=64).tolist()
        df["embeddings"] = embeddings
        print('Done calculating embeddings')
        return df

    def search_embeddings(self, df, query, n=3, pprint=True):
        query_embedding = self.agent.get_embedding([query]).tolist()[0]
        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
        
        results = df.sort_values("similarity", ascending=False, ignore_index=True)
        # make a dictionary of the the first three results with the page number as the key and the text as the value. The page number is a column in the dataframe.
        results = results.head(n)
        global sources 
        sources = []
        for i in range(n):
            # append the page number and the text as a dict to the sources list
            sources.append({'Page '+str(results.iloc[i]['page']): results.iloc[i]['text'][:150]+'...'})
        print(sources)
        return results.head(n)
    
    def create_prompt(self, df, user_input):
        result = self.search_embeddings(df, user_input, n=3)
        print(result)
        prompt = """You are a large language model whose expertise is reading and summarizing scientific papers. 
        You are given a query and a series of text embeddings from a paper in order of their cosine similarity to the query.
        You must take the given embeddings and return a very detailed summary of the paper that answers the query.
            
            Given the question: """+ user_input + """
            
            and the following embeddings as data: 
            
            1.""" + str(result.iloc[0]['text']) + """
            2.""" + str(result.iloc[1]['text']) + """
            3.""" + str(result.iloc[2]['text']) + """

            Return a detailed answer based on the paper:"""

        print('Done creating prompt')
        return prompt

    def gpt(self, prompt):
        print('Sending request to GPT-3')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        r = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.4, max_tokens=1500)
        answer = r.choices[0]['text']
        print('Done sending request to GPT-3')
        response = {'answer': answer, 'sources': sources}
        return response

    def reply(self, prompt):
        print(prompt)
        prompt = self.create_prompt(df, prompt)
        return self.gpt(prompt)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/process_pdf", methods=['POST'])
def process_pdf():
    print("Processing pdf")
    file = request.data
    pdf = PdfReader(BytesIO(file))
    chatbot = Chatbot()
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("Done processing pdf")
    return {'key': ''}

@app.route("/download_pdf", methods=['POST'])
def download_pdf():
    chatbot = Chatbot()
    url = request.json['url']
    r = requests.get(str(url))
    print(r.headers)
    pdf = PdfReader(BytesIO(r.content))
    paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text)
    df = chatbot.calculate_embeddings(df)
    print("Done processing pdf")
    return {'key': ''}

@app.route("/reply", methods=['POST'])
def reply():
    chatbot = Chatbot()
    query = request.json['query']
    query = str(query)
    prompt = chatbot.create_prompt(df, query)
    response = chatbot.gpt(prompt)
    print(response)
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
