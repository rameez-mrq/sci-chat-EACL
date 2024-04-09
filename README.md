# Usage
1. Run ```python start_server.py```
2. Open `scichat_w_topic.html` in Chrome. (check if `base_url` in HTML (line 802) is same as the flask server address.)
3. Now you can start chat with bots.

# Requirements

- python=3.8
- transformers
- pytorch
- tqdm
- flask
- flask-cors

# Files

## MTurk user interface files

The contents of these two files can be  directly copied and pasted to MTurk.
- `scichat_mturk_template_w_topic.html`: HMTL code of free topic (please check the comments at line 5 and line 769)

## Server
- `start_server.py`: it runs a Python server. 
- `utils.py`: it is the code of dialogue models.

## Other files
- `degraded_random_responses_filtered.txt`: response candidates of the qc model.
- `topics_podcast.txt`: extracted conversation topics from the podcasts.
