import pandas as pd
import re
import string
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
import stylecloud
from IPython.display import display

def make_wordcloud(
    main_text: pd.Series,
    color_palette: list,
    comparison_text: pd.Series = None,
    ignore_words = [],
    combine_words = [],
    out_words: int = 30,
    title: str = 'Title',
    subtitle: str = 'Subtitle',
    title_size = 30,
    subtitle_size = 18,
    filepath: str = './wordcloud.png',
    rng_seed: int = 123
):

    # list of stopwords taken from NLTK
    stopwords = pd.read_csv('stopwords.csv')['0'].tolist() # in newsletters/ directory
    def remove_stopwords(input_txt):
        words = input_txt.lower().split()
        noise_free_words = [word for word in words if word not in stopwords] 
        noise_free_text = " ".join(noise_free_words) 
        return noise_free_text

    def convert_to_tfidf(text_series):
        text_series = text_series.apply(lambda x: re.sub(f'[{string.punctuation}]', ' ', x))
        text_series = text_series.apply(lambda x: re.sub(r'\d+', '', x))
        text_series = text_series.apply(lambda x: remove_stopwords(x))
        for i, j in combine_words:
            text_series = text_series.apply(lambda x: x.replace(i, j))
        vectorizer_cons = TfidfVectorizer()
        tfidf_transposed = vectorizer_cons.fit_transform(text_series)
        tfidf_transposed = pd.DataFrame(tfidf_transposed.toarray(), columns = vectorizer_cons.get_feature_names())
        tfidf_series = tfidf_transposed.T.sum(axis=1)
        return tfidf_series

    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    
    main_tfidf = convert_to_tfidf(main_text)
    if comparison_text is not None:
        comparison_tfidf = convert_to_tfidf(comparison_text)
    else:
        comparison_tfidf = pd.Series([0]*len(main_tfidf))

    df = pd.concat([main_tfidf, comparison_tfidf], axis = 1).reset_index()
    df.columns = columns=['word', 'main_tfidf', 'comparison_tfidf']
    df['diff_tfidf'] = df.main_tfidf - df.comparison_tfidf

    sub_index = (~df.word.isin(ignore_words)) & (df.main_tfidf.notna()) & (df.comparison_tfidf.notna())
    df = df[sub_index]
    df.sort_values('diff_tfidf', ascending = False, ignore_index = True)[:out_words][['word', 'diff_tfidf']].to_csv(f'tfidf.csv', index = False)

    stylecloud.gen_stylecloud(
        file_path = 'tfidf.csv',
        icon_name = "fas fa-square-full",
        palette = 'colorbrewer.diverging.Spectral_11',
        colors = color_palette,
        background_color = 'white',
        gradient = 'horizontal',
        size = (460, 460), # set to match standard plot height: 460 + 140 = 600
        random_state = rng_seed,
        output_name = f"{filepath}_untitled.png"
    )
    
    img = Image.open(f"{filepath}_untitled.png")

    # add margins to make room for titles and watermark - set to match standard plot height
    img = add_margin(
        img,
        top = 100,
        bottom = 40,
        right = 0,
        left = 0,
        color = (255, 255, 255)
    )

    # add title and subtitle
    d1 = ImageDraw.Draw(img)
    title_font = ImageFont.truetype('Roboto-Medium.ttf', title_size)
    sub_font = ImageFont.truetype('Roboto-Regular.ttf', subtitle_size)
    d1.text(
        (0, 10),
        title, 
        font = title_font,
        fill = '#2D426A'
    )
    d1.text(
        (0, 60),
        subtitle, 
        font = sub_font,
        fill = '#899499'
    )

    !rm '{filepath}_untitled.png'
    !rm 'tfidf.csv'

    img.save(f'{filepath}.png', quality = 100)
    display(Image.open(f'{filepath}.png'))
    
    return None
