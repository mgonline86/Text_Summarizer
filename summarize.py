# coding=utf8

import nltk
import re
import heapq

def summarize_text(text, summary_percent):
        
    # Getting Packages if not installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Removing special characters and digits
    formatted_text = re.sub('[^\u0600-\u06FFa-zA-Z0-9]', ' ', text )
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    # Converting Text To Sentences
    # sentence_list = nltk.sent_tokenize(text)
    sentence_list = re.split(r'([.,،])', text)
    for i,e in enumerate(sentence_list):
        if e == '':
            del sentence_list[i]
            continue
        if i+1 < len(sentence_list):
            if sentence_list[i+1] == '.' or sentence_list[i+1] == ',' or sentence_list[i+1] == '،':
                sentence_list[i] += sentence_list[i+1]
                del sentence_list[i+1]

    # Find Weighted Frequency of Occurrence
    stopwords = nltk.corpus.stopwords.words(['arabic','english'])

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())
    # maximum_frequncy_word = max(word_frequencies.keys())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Calculating Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # Setting max number of sentences in the summary
    max_sent_count = int(round(summary_percent * len(sentence_list)))

    summary_sentences = heapq.nlargest(max_sent_count, sentence_scores, key=sentence_scores.get)

    # Sorting Sentences in correct order
    sorted_summary_sentences = [ele for x in sentence_list for ele in summary_sentences if ele == x]
    highlighted_summary = []
    for sent in sentence_list:
        if sent in sorted_summary_sentences:
            highlighted_summary.append('<span class="highlighted">' + sent + '</span>')
        else:
            highlighted_summary.append(sent)

    summary = ' '.join(sorted_summary_sentences)
    higlighted_summary = ' '.join(highlighted_summary)

    converted_data = {
        'summary' : summary,
        'higlighted_summary' : higlighted_summary,
    }    
    return converted_data