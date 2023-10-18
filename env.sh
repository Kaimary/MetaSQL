apt-get update
apt-get install -y build-essential
apt-get clean
pip install allennlp==2.9.3
pip install dgl-cu113==0.9.1 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements.txt
pip install protobuf==3.20
python -m spacy download en_core_web_sm
echo "en_core_web_sm downloaded"
python -c "import nltk;nltk.download('punkt');nltk.download('stopwords');nltk.download('wordnet')"
python -c "import stanza; stanza.download('en')"
mkdir output
ln -s ../pretrained_models pretrained_models
ln -s ../saved_models saved_models
ln -s ../model.bin ./