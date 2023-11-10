pip install allennlp==2.9.3
pip install dgl-cu113==0.9.1 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install transformers==4.18.0
pip install huggingface_hub==0.10.1
pip install protobuf==3.20
python -m spacy download en_core_web_sm
echo "en_core_web_sm downloaded"
python -c "import nltk;nltk.download('punkt');nltk.download('stopwords');nltk.download('wordnet')"
python -c "import stanza; stanza.download('en')"
mkdir output
