# wordcloud
Function for making a TF-IDF-based word cloud in python

make_wordcloud() takes the following arguments and creates, saves and displays a word cloud:

-main_text: a pandas Series containing the texts in which to measure word frequencies

-color_palette: a list of hex colors for word shades

-comparison_text (optional): a pandas Series containing the "baseline/control" texts from which to measure frequencies in main_text

-ignore_words (optional): a list of words to ignore in addition to NLTK's standard stopwords (see stopwords.csv)

-combine_words (optional): a list of 2-tuples to combine - e.g., [('annoyed', 'annoying')] would combine the contributions of those words into 'annoying'

-out_words (optional, default = 30): the number of words in the cloud

-title, subtitle, title_size, subtitle_size (optional, defaults = Shutterbutton defaults) - self-explanatory

-filepath (optional, default = './wordcloud'): filepath for the output png

-rng_seed (optional, default = 123): the random_state for the wordcloud function (the positioning is non-deterministic)
