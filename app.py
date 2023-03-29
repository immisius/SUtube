from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi
app = Flask(__name__)
# Hugging Faceの事前学習済みモデルを使用して、字幕を要約する    
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Hugging Faceの事前学習済みモデルをロードする
tokenizer = AutoTokenizer.from_pretrained("tsmatz/mt5_summarize_japanese")

model = AutoModelForSeq2SeqLM.from_pretrained("tsmatz/mt5_summarize_japanese")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    video_id = request.form['video_id']

    captions=YouTubeTranscriptApi.get_transcript(video_id,languages=['ja'])
    # captionをひとまとまりにする
    caption=''
    for i in range(len(captions)):
        caption+=captions[i]['text']
    inputs = tokenizer.encode(caption, return_tensors="pt", max_length=10000, truncation=True)
    print(caption[:100])
    outputs = model.generate(inputs, max_length=1500,min_length=300, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary=tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(summary[:100])
    return render_template('result.html', caption=caption, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)






