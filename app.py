from flask import Flask, render_template
# from My_QA_model import answer_question
from flask import Flask, render_template, request, redirect, url_for
# import torch
# from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from information_ret import get_most_similar_docs
# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message
import pandas as pd
import random
from googletrans import Translator


translator = Translator()

def translate_to_hebrew(text):
    translated = translator.translate(text, dest='he').text
    return translated

app.config['MAIL_SERVER'] = "smtp.office365.com" # smtp.office365.com
app.config['MAIL_POST'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'women_org1122@hotmail.com'
app.config['MAIL_PASSWORD'] = '123456women'

mail = Mail(app)
app.secret_key = 'womenDiseaseHELP'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fibro')
def fibro():
    plotly_urls = [
        "https://plotly.com/~transparentwomenproject/195/",
        "https://plotly.com/~transparentwomenproject/197/",
        "https://plotly.com/~transparentwomenproject/199/",
        "https://plotly.com/~transparentwomenproject/201/",
        "https://plotly.com/~transparentwomenproject/207/",
        "https://plotly.com/~transparentwomenproject/205/",
        "https://plotly.com/~transparentwomenproject/209/",
        "https://plotly.com/~transparentwomenproject/213/",
        "https://plotly.com/~transparentwomenproject/221/",
        "https://plotly.com/~transparentwomenproject/217/",
        "https://plotly.com/~transparentwomenproject/219/"
    ]
    return render_template('fibro.html', plotly_urls=plotly_urls)

@app.route('/eds')
def eds():
    plotly_urls = [
    "https://plotly.com/~hannabaw12121313/42/", # question 19
    "https://plotly.com/~hannabaw12121313/44/", # question 77
    "https://plotly.com/~hannabaw12121313/48/", # quetion 40
    "https://plotly.com/~hannabaw12121313/46/", # question 54
    "https://plotly.com/~hannabaw12121313/39/" # question 71
    ]
    return render_template('eds.html', plotly_urls=plotly_urls)



@app.route('/google-form')
def google_form():
    return render_template('google_form.html')




@app.route('/qa_page', methods=['GET', 'POST'])
def qa_page():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        topic = request.form['topic']
        if topic == "eds_disease":
            whatToKnow = request.form['chooseToKnow']
            if whatToKnow == 'symptoms':
                answer = get_most_similar_docs(question, 'symp', topic)
                
            if whatToKnow == 'types':
                answer = get_most_similar_docs(question, 'types', topic)
            
    if answer is None:
        answer = "אנחנו פה כדי לספק עבורך מידע להעלאת מודעתך, תזכרי שתמיד אנחנו איתך."
    return render_template('qa_page.html', answer=translate_to_hebrew(answer))

@app.route('/contact', methods = ['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        customer_mail = request.form['customer_email']
        message = request.form['message']
        sender_mail = request.form['sender_mail']

        msg = Message(
            "New message from your WHLWD web",
            sender=app.config['MAIL_USERNAME'],
            recipients=[customer_mail]
        )

        msg.body = f"Mail from : {name}\n\n The sender mail: {sender_mail}\n\n{message}"

        try:
            mail.send(msg)
            flash("Your message has been sent! We will contact you soon.")
        except:
            flash("sorry there is an error, come back soon!")

        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/knowus')
def knowUS():
    return render_template('knowus.html')






@app.route("/stories", methods=["GET", "POST"])
def stories():
    if request.method == "POST":
        stories = pd.read_csv('data/generateText.csv')
        topic = request.form["topic"]
        print(topic)
        if topic == 'Eds':
            eds_stories = list(stories[stories['label'] == 'eds'].iloc[:,0])
            text = random.choice(eds_stories)
            return render_template("stories.html", text=translate_to_hebrew(text))
        else:
            text = 'We have no stories about this topic yet'
            return render_template("stories.html", text=text)
    else:
        return render_template("stories.html")






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

