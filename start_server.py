import logging
from systemd import journal
from flask import Flask, jsonify, request, abort, render_template
from flask_cors import CORS
from utils import DialogueModels, PersonaManager
import torch


# debug_mode = True
debug_mode = False
dialogue_models = DialogueModels() 
persona_manager = PersonaManager()

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
journaldHandler = journal.JournaldLogHandler()
logger.addHandler(journaldHandler)

logger.error('TESTING LOGGING')

def clean_text(text):
    # remove special symbols
    user_input = text.replace("#"," ").replace("^"," ").replace("@"," ").replace("|"," ").strip()
    user_input = ' '.join(user_input.split())
    return user_input

def history_txt_to_list(history_txt):
    history = history_txt.strip().split("###")
    history = [e[3:].strip() for e in history if e != ""]
    return history


@app.route('/')
def indexpage():
    return render_template('scichat_w_topic.html')

@app.route('/api/model/<model>/interact/',methods=["GET","POST"])
def interact(model):
    if request.method == "POST":
        form_values = request.form
    else:
        form_values = request.args
    logger.info('form_values: %s', form_values)
    
    user_input = clean_text(form_values.get("text",""))
    if user_input == "":
        return abort(404, description="Invalid user input")
    
    history = history_txt_to_list(form_values.get("history",""))
    topic = form_values.get("topic","")
    input_data = {
        "model": model,
        "user_input": user_input,
        "history": history,
        "personas": topic,
    }
    print("Input Data:", input_data)
    response = dialogue_models.get_response(input_data)
    full_response = {
        "user_input": user_input,
        "response": response,
    }
    return [full_response]
    

@app.route('/api/icebreaker',methods=["GET","POST"])
def rand_topic():
    return jsonify({
        "topic": persona_manager.get_single_persona(),
    })


@app.route('/api/topic_change',methods=["GET","POST"])
def rand_topics_multiple():
    return jsonify({
        "topics": persona_manager.get_persona(),
    })


@app.route('/error')
def error_route():
    print("error")
    return abort(501)

def main():
    '''
    Do not enable debug mode for real MTurk deployment.
    '''
    
    app.run(
        host='0.0.0.0', 
        threaded=True,
        debug=debug_mode,
    )
if __name__ == '__main__':
    main()
