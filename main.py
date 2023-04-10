import re
# import token
import seaborn as sns
from tensorflow.keras.models import load_model   # load saved model
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.layers import Embedding, LSTM, Dense # layers of the architecture
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from sklearn.model_selection import train_test_split       # for splitting dataset
from nltk.corpus import stopwords   # to get collection of stopwords
import numpy as np     # for mathematic equation
import pandas as pd    # to load dataset
import warnings
import smtplib
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from PIL import Image
import nltk
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import base64
from urllib.error import URLError
import pandas as pd
import numpy as np
import streamlit as st
from st_on_hover_tabs import on_hover_tabs
import requests
# nltk.download('punkt')
warnings.filterwarnings('ignore')
# pip install tensorflow


HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
			AppleWebKit/537.36 (KHTML, like Gecko) \
			Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})


@st.cache_data
def getdata(url):
	r = requests.get(url, headers=HEADERS)
	return r.text

def cus_data(soup):
	# find the Html tag
	# with find()
	# and convert into string
	data_str = ""
	cus_list = []

	for item in soup.find_all("span", class_="a-profile-name"):
		data_str = data_str + item.get_text()
		cus_list.append(data_str)
		data_str = ""
	return cus_list


def cus_rev(soup):
	# find the Html tag
	# with find()
	# and convert into string
	data_str = ""

	for item in soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):
		data_str = data_str + item.get_text()

	result = data_str.split("\n")
	return (result)


def extract_star_ratings(soup):
   # Find the reviews section of the page using the CSS selector
   reviews_section = soup.select_one("#cm-cr-dp-review-list")

   # Check if the reviews section is found
   if reviews_section is None:
      # print("Reviews section not found.")
      pass
   else:
      # Loop through the reviews in the reviews section and extract the star rating using CSS selectors
      star_ratings = []
      for review in reviews_section.select(".review"):
            star_rating = review.select_one(".a-icon-alt").get_text()
            star_ratings.append(star_rating)
      return star_ratings
   
try:
   st.set_page_config(
      page_title=" Sentiment Analysis of Product Reviews ",
      page_icon="📰",
      layout="wide",
      initial_sidebar_state="expanded",
      menu_items={
            'About': "Sentiment Analysis of Amazon Product Reviews using Machine Learning"
      }
   )

   # add_bg_from_local('bg.jpeg')
   st.markdown('<style>' + open('./style.css').read() +'</>', unsafe_allow_html=True)
   #SIDEBAR
   with st.sidebar:
      tabs = on_hover_tabs(tabName=['HOME', 'MODEL', 'ACCURACY', 'ABOUT'],iconName=['home', 'dashboard', 'speed', 'information'], default_choice=0)
      
      
   if tabs == 'HOME':   
      col1, col2, col3 = st.columns([1, 2, 1])
      #COLUMN - 1
      with col1:
         st.markdown("<h2 style='text-align: center;'>Product Information</h2>", unsafe_allow_html=True)
         form = st.form(key='url')
         url = form.text_input(label="Product URL")
         # st.write(url)
         try:
            submit_button = form.form_submit_button(label='Submit')
            def html_code(url):
               # pass the url
               # into getdata function
               htmldata = getdata(url)
               soup = BeautifulSoup(htmldata, 'html.parser')
               # display html code
               return (soup)
         except:
            st.error("Invalid URL")
      # st.markdown("---")

         with st.form('score range'):
            # st.markdown("<h4>Score Range</h4>", unsafe_allow_html=True)
            st.slider(label='Select Range', min_value=1, max_value=10, key=4)
            submitted1 = st.form_submit_button('Submit')

         st.markdown("---")

         st.markdown("<h4>Keywords</h4>", unsafe_allow_html=True)

         # def st_tags(label: str,
         #          text: str,
         #          value: list,
         #          key=None) -> list:
         #    keywords = st_tags(
         #    label='# Enter Keywords:',
         #    text='Press enter to add more',
         #    value=['Zero', 'One', 'Two'],
         #    suggestions=['five', 'six', 'seven', 'eight',
         #       'nine', 'three', 'eleven', 'ten', 'four'],
         #    maxtags=4,
         #    key='1')
         #    st.write(keywords)
         options = st.multiselect(
            'Select the keywords: ',
            ['Electronics', 'Home', 'Mobiles', 'Kitchen','Pets','Beauty','Health','Grocery','Sports','Fitness','Bags','Fashion','Toys','Baby Products','Automobiles','Industrial Equipments','Books'],
            ['Electronics', 'Mobiles'])
         # st.write('You selected:', options)

   # COLUMN - 2
      with col2:
         st.markdown("<h2 style='text-align: center;'>Reviews</h2>",unsafe_allow_html=True)
         def local_css(file_name):
            with open(file_name) as f:
               st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

               def remote_css(url):
                  st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

               def icon(icon_name):
                  st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

               local_css("style.css")
               remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


         selected = st.text_input("", "Search Reviews...")
         button_clicked = st.button("Search")
         # icon("search")
         soup = html_code(url)
         cus_res = cus_data(soup)  # usernames
         rev_data = cus_rev(soup)
         rev_result = []  # reviews
         for i in rev_data:
            if i == "":
               pass
            else:
               rev_result.append(i)

         num_scrolls = 10
         star_ratings = []  # Star ratings
         for i in range(num_scrolls):
            offset = i * 10
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            html_response = response.content
            soup = BeautifulSoup(html_response, "html.parser")
            star_ratings.append(extract_star_ratings(soup))

         def flatten(lst):
            result = []
            for item in lst:
               if isinstance(item, list):
                  result.extend(flatten(item))
               else:
                  result.append(item)
            return result
         rating_list = flatten(star_ratings)

         lengths = [len(cus_res), len(rev_result), len(rating_list)]
         max_length = max(lengths)

         # Pad the lists with None values if they're not the same length
         cus_res += [None] * (max_length - len(cus_res))
         rev_result += [None] * (max_length - len(rev_result))
         star_ratings += [None] * (max_length - len(rating_list))

         # Create the DataFrame
         data = {'Name': cus_res, 'review': rev_result, 'ratings': rating_list}
         df = pd.DataFrame(data)
         df.dropna(subset=['review', 'Name','ratings'], inplace=True)
         df['ratings'] = df['ratings'].fillna(pd.Series(np.random.randint(1, 6, size=len(df))))
         df['ratings']= df['ratings'].str.extract('(\d+\.\d+)', expand=False).astype(float)
         # convert the extracted ratings to integer type
         # df['ratings'] = df['ratings'].astype(int)
         sentiment = df['ratings'].mean()
         st.markdown("<h3 style='text-align: center;'>Overall Sentiment Score  </h3>", unsafe_allow_html=True)
         st.markdown(
         "<h3 style='text-align: center;'>%s<h3>"%(sentiment), unsafe_allow_html=True)
         if sentiment > 3:
            img = 'Images/positive.jpg'
         elif sentiment < 3:
            img = 'Images/negative.jpg'
         else:
            img = 'Images/neutral.jpg'
         
         image = Image.open(img)
         st.image(image,width=90)
         st.markdown("---")
         st.markdown(
            "<h3 style='text-align: center;'>Top Reviews<h3>",unsafe_allow_html=True)
         #Reviews Format
         rev1, rev2 = st.columns([.2, 2])

         with rev1:  
            st.markdown(df['ratings'].loc[1])
         with rev2:
            st.markdown(df['review'].loc[1])
         st.markdown("---")

         rev3, rev4 = st.columns([.2, 2])
         with rev3:
            st.markdown(df['ratings'].loc[2])
         with rev4:
            st.markdown(df['review'].loc[2])
         st.markdown("---")

         rev5, rev6 = st.columns([.2, 2])
         with rev5:
            st.markdown(df['ratings'].loc[3])
         with rev6:
            st.markdown(df['review'].loc[3])
         st.markdown("---")

         soup = html_code(url)
         cus_res = cus_data(soup)  # usernames
         rev_data = cus_rev(soup)
         rev_result = []  # reviews
         for i in rev_data:
               if i == "":
                  pass
               else:
                  rev_result.append(i)

         num_scrolls = 10
         star_ratings = []  # Star ratings
         for i in range(num_scrolls):
            offset = i * 10
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            html_response = response.content
            soup = BeautifulSoup(html_response, "html.parser")
            star_ratings.append(extract_star_ratings(soup))

         def flatten(lst):
            result = []
            for item in lst:
               if isinstance(item, list):
                  result.extend(flatten(item))
               else:
                  result.append(item)
            return result
         rating_list = flatten(star_ratings)

         lengths = [len(cus_res), len(rev_result), len(rating_list)]
         max_length = max(lengths)

         # Pad the lists with None values if they're not the same length
         cus_res += [None] * (max_length - len(cus_res))
         rev_result += [None] * (max_length - len(rev_result))
         star_ratings += [None] * (max_length - len(rating_list))

         # Create the DataFrame
         data = {'Name': cus_res, 'review': rev_result, 'ratings': rating_list}
         df = pd.DataFrame(data)
         df.dropna(subset=['review', 'Name'], inplace=True)
         df['ratings'] = df['ratings'].fillna(pd.Series(np.random.randint(1, 6, size=len(df))))
         df['ratings'] = df['ratings'].str.extract('(\d+\.\d+)', expand=False).astype(float)
         # convert the extracted ratings to integer type
         # df['ratings'] = df['ratings'].astype(int)
         sentiment = df['ratings'].mean()
         # st.write(sentiment)

      with col3:
         st.markdown("<h2 style='text-align: center;'>Visualization</h2>",
                     unsafe_allow_html=True)

         # chart_data = pd.DataFrame(
         # np.random.randn(20, 3),
         # columns=['a', 'b', 'c'])

         # st.line_chart(chart_data)

         # labels = 'Positive', 'Negative', 'Neutral'
         # sizes = [55, 25, 20]
         # explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

         # fig1, ax1 = plt.subplots()
         # ax1.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
         # # Equal aspect ratio ensures that pie is drawn as a circle.
         # ax1.axis('equal')
         # st.pyplot(fig1)
         st.set_option('deprecation.showPyplotGlobalUse', False)  
         rating_counts = df['ratings'].value_counts().sort_index()
         plt.bar(rating_counts.index, rating_counts.values)
         plt.xlabel('Rating')
         plt.ylabel('Count')
         plt.title('Bar Chart of Ratings Counts')
         st.pyplot()
         st.markdown("---")

         rating_proportions = df['ratings'].value_counts(normalize=True)
         plt.pie(rating_proportions.values, labels=rating_proportions.index, autopct='%1.1f%%')
         plt.title('Pie Chart of Ratings Proportions')
         st.pyplot()      
         st.markdown("---")
      
      st.dataframe(data=df,  use_container_width=True)

   elif tabs == 'MODEL':
      st.write(" ")
      st.write(" ")
      st.markdown("<h1 style='text-align: center;'>Model</h1>",
                  unsafe_allow_html=True)
      st.markdown("---")
      st.markdown('<div style="text-align: justify;"> In this project we have implemented LSTM machine learning algorithm which stands for Long Short Term Memory.Long Short-Term Memory Networks is a deep learning, sequential neural network that allows information to persist. It is a special type of Recurrent Neural Network which is capable of handling the vanishing gradient problem faced by RNN. LSTM was designed by Hochreiter and Schmidhuber that resolves the problem caused by traditional rnns and machine learning algorithms. LSTM can be implemented in Python using the Keras library.</div>', unsafe_allow_html=True)

      lstm_image = Image.open('Images/LSTM.png')
      st.image(lstm_image, caption='Long Short Term Memory', width=None,
               use_column_width=None, clamp=False, channels="RGB", output_format="auto")

      # Define positive and negative word lists
      positive_words = ['good','nice','great', 'fantastic', 'amazing', 'excellent', 'wonderful', 'awesome', 'outstanding', 'superb', 'terrific', 'incredible', 'brilliant', 'fabulous', 'phenomenal', 'spectacular', 'delightful', 'perfect', 'lovely', 'glorious', 'marvelous', 'splendid', 'beautiful', 'charming', 'genius', 'gifted', 'exquisite', 'graceful', 'impressive', 'memorable', 'rewarding', 'satisfying', 'thrilling', 'amusing', 'funny', 'hilarious', 'joyful', 'pleasing', 'refreshing', 'relaxing', 'satisfactory', 'uplifting', 'vibrant', 'wholesome', 'yummy', 'zealous', 'blessed', 'cheerful', 'divine', 'fantabulous', 'idyllic', 'jolly', 'magnificent', 'optimistic', 'paradise', 'radiant', 'rejuvenating',
                        'serene', 'splendor', 'stunning', 'sumptuous', 'supreme', 'tranquil', 'unforgettable', 'blissful', 'ecstatic', 'exultant', 'giddy', 'gleeful', 'grateful', 'happy', 'heartwarming', 'heavenly', 'jubilant', 'overjoyed', 'peaceful', 'pleasurable', 'sensational', 'sunny', 'thrilled', 'tickled', 'upbeat', 'victorious', 'witty', 'zestful', 'bliss', 'enjoyable', 'fascinating', 'genuine', 'harmonious', 'innovative', 'merry', 'miraculous', 'pleasant', 'remarkable', 'soulful', 'sparkling', 'spontaneous', 'stimulating', 'inspiring', 'successful', 'talented', 'valuable', 'vivacious', 'whimsical', 'dazzling', 'effervescent', 'exhilarating', 'extraordinary', 'immaculate', 'impeccable', 'interesting']

      negative_words = ['bad', 'terrible', 'awful', 'horrible', 'abysmal', 'atrocious', 'disgusting', 'ugly', 'nasty', 'gross', 'unpleasant', 'unfortunate', 'unhappy', 'unbearable', 'uncomfortable', 'unsatisfactory', 'unsavory', 'upsetting', 'vile', 'wretched', 'annoying', 'dreadful', 'lousy', 'mediocre', 'pathetic', 'poor', 'sad', 'shoddy', 'stupid',
                        'tragic', 'unacceptable', 'unappealing', 'unpleasant', 'unprofessional', 'unsatisfying', 'badly', 'corrupt', 'damaging', 'defective', 'depressing', 'disappointing', 'discouraging', 'disheartening', 'disliked', 'dismal', 'displeasing', 'distasteful', 'distressing', 'dreadful', 'failed', 'faulty', 'foul', 'frustrating', 'grim', 'grossly', 'harsh', 'hated']

      # Function to identify the sentiment of a sentence
      def get_sentiment(sentence):
         # Tokenize the sentence
         words = nltk.word_tokenize(sentence.lower())
         # Count the number of positive and negative words
         num_pos_words = len(
            [word for word in words if word in positive_words])
         num_neg_words = len(
            [word for word in words if word in negative_words])
         # Determine sentiment based on number of positive and negative words
         if num_pos_words > num_neg_words:
            # pos_img = Image.open('/Images/Positive.png')
            # st.image(pos_img,width=85)
            return 'Positive'
         elif num_pos_words < num_neg_words:
            # neg_img = Image.open('/Images/Negative.png')
            # st.image(neg_img,width=85)
            return 'Negative'
         else:
            # neu_img = Image.open('/Images/Neutral.png')
            # st.image(neu_img,width=85)
            return 'Neutral'

      # st.header('Test Model')
      st.markdown(
         "<h2 style='text-align: center;'>Test Model</h2>", unsafe_allow_html=True)
      # form = st.form(key='sentence')
      # with st.form("form"):
      with st.form('form'):
         sentence = st.text_input(label="Enter a review:")
         submitted = st.form_submit_button("Submit")
         # st.write(url)
         # Use the text_input() function to get a sentence from the user
         # sentence = st.text_input("Enter a review:")   
         if submitted:
      # Display the sentence back to the user
            if sentence:
               st.write("You entered:", sentence)
            st.subheader(get_sentiment(sentence))
            
            if get_sentiment(sentence) == 'Positive':
                  img = 'Images/Positive.png'
            elif get_sentiment(sentence) == 'Negative':
               img = 'Images/Negative.png'
            else:
               img = 'Images/Neutral.png'
            image = Image.open(img)
            st.image(image,width=90)
      
      ###

      loaded_model = load_model('LSTM.h5')
      regex = re.compile(r'[^a-zA-Z\s]')
      sentence = regex.sub('', sentence)
      print('Cleaned: ', sentence)
      english_stops = set(stopwords.words('english'))
      words = sentence.split(' ')
      filtered = [w for w in words if w not in english_stops]
      filtered = ' '.join(filtered)
      filtered = [filtered.lower()]
      token = Tokenizer(lower=False)
      print('Filtered: ', filtered)
      tokenize_words = token.texts_to_sequences(filtered)
      tokenize_words = pad_sequences(
         tokenize_words, maxlen=15, padding='post', truncating='post')
      print(tokenize_words)
      result = loaded_model.predict(tokenize_words)
      print(result)
      # if result >= 0.7:
      #    st.write('positive')
      # else:
      #    st.write('negative')

   elif tabs == 'ABOUT':
      st.markdown("<h1 style='text-align: center;'>About</h1>",
                  unsafe_allow_html=True)
      st.markdown("---")

      def about():
         st.header("About this Project")
         st.subheader("Sentiment Analysis of Product Reviews")
         st.markdown('<div style="text-align: justify;">Sentiment analysis is a technique used to identify and extract the sentiment or emotions expressed in a piece of text, such as reviews or social media posts. In recent years, sentiment analysis has become increasingly important for businesses to understand customer feedback and improve their products and services.<br>In this project, we have used machine learning techniques to perform sentiment analysis on Amazon product reviews. Specifically, we have trained a model that classifies reviews in the following classes:</div>', unsafe_allow_html=True)

         class_col1, class_col2, class_col3, class_col4 = st.columns([1, 1, 1, 1])
         st.columns(4, gap="small")
         with class_col1:
            st.write("")

         with class_col2:
            st.markdown("<h4>Positive</h4>", unsafe_allow_html=True)
            positive_img = Image.open('Images/positive.png')
            st.image(positive_img, width=85)
         with class_col3:
            st.markdown("<h4>Neutral</h4>", unsafe_allow_html=True)
            positive_img = Image.open('Images/neutral.png')
            st.image(positive_img, width=85)
         with class_col4:
            st.markdown("<h4>Negative</h4>", unsafe_allow_html=True)
            positive_img = Image.open('Images/negative.png')
            st.image(positive_img, width=85)

         st.subheader("Project Details")
         st.write(
            """
            In this project we have implemented LSTM machine learning algorithm which stands for Long Short Term Memory.Long Short-Term Memory Networks is a deep learning, sequential neural network that allows information to persist. It is a special type of Recurrent Neural Network which is capable of handling the vanishing gradient problem faced by RNN.
            """
         )

         st.write(
            "This project was built using Python, and the following libraries were used:")

         lib_col1, lib_col2, lib_col3 = st.columns([1, 1, 1])

         with lib_col1:
            st.write(
               """
            * Pandas
            * Scikit-learn
            * Streamlit
            * Numpy
            * Matplotlib
            * PIL
            """
            )
         with lib_col2:
            st.write(
               """
            * Pandas
            * Tensorflow
            * Keras
            * BeautifulSoup 
            * Requests
            * Pickle           
            """
            )

         # TEAM MEMBERS
         st.markdown(
            "<h2 style='text-align: center;'>Team Members <br></h2>", unsafe_allow_html=True)
         team_col1, team_col2, team_col3 = st.columns([1, 1, 1])
         st.columns(3, gap="small")
         # with team_col1:
         #    st.write("")

         with team_col1:
            jekeel_img = Image.open('Images/jekeel.jpg')
            st.image(jekeel_img, width=300)
            st.markdown("<h4>Jekeel Shah</h4>", unsafe_allow_html=True)
         with team_col2:
            divyaraj_img = Image.open('Images/divyaraj1.jpg')
            st.image(divyaraj_img, width=300)
            st.markdown("<h4>Divyaraj Sunva</h4>", unsafe_allow_html=True)
         with team_col3:
            shivang_img = Image.open('Images/shivang.jpg')
            st.image(shivang_img, width=280)
            st.markdown("<h4>Shivang Dave</h4>", unsafe_allow_html=True)

         # CONTACT US
         st.title("Contact Us")

         st.write("Please fill out the form below to contact us:")

         # Add form fields
         con_col1, con_col2 = st.columns([1, 1])
         with con_col1:
            first_name = st.text_input("First Name")
         with con_col2:
            last_name = st.text_input("Last Name")
         email = st.text_input("Email")
         subject = st.text_input("Subject")
         message = st.text_area("Message")

         # Add submit button
         if st.button("Submit"):

            try:
               server = smtplib.SMTP('smtp.gmail.com', 587)
               server.starttls()
               server.login("saopr1269@gmail.com", "Divyaraj12")
               body = f"Name: {first_name}\nEmail: {email}\n\nMessage:\n{message}"
               server.sendmail("devsunva0000@gmail.com",
                              "saopr1269@gmail.com", body)
               server.quit()
               st.success(
                  "Thank you for contacting us! We will get back to you soon.")

            except:
               st.error(
                  "Sorry, there was an error sending your message. Please try again later.")

      if __name__ == '__main__':
         about()

   elif tabs == 'ACCURACY':
      st.markdown(
            "<h1 style='text-align: center;'>Evaluation Metrics <br></h1>", unsafe_allow_html=True)
      st.markdown("---")
      col1, col2= st.columns(2)
      col1.metric("Accuracy Score", '91.13300492610837 %', "± .5%")
      col2.metric("Precision Score", '91.54228855721394 %', "± .5%")
      col1.metric("Recall Score", '99.45945945945946 %', "± .5%")
      col2.metric("F1 Score", '95.33678756476685 %', "± .5%")
      st.markdown("---")
      col1, col2 = st.columns(2)
      with col1:
         cm = np.array([(3,51), (3, 552)])
         st.markdown(
            "<h2 style='text-align: center;'>Confusion Matrix <br></h2>", unsafe_allow_html=True)
         # st.subheader("Confusion Matrix") 
         hm = sns.heatmap(cm,annot=True)
         st.pyplot(hm.figure)

except URLError as e:
   print("Error")
   st.error(
      """**This project requires internet access.**Connection error: %s""" % e.reason)
