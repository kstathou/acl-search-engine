import streamlit as st
import pickle
import altair as alt
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

@st.cache
def read_data():
    with open('acl_data.pickle', 'rb') as h:
        return pickle.load(h)

@st.cache
def read_author_data():
    with open('author_data.pickle', 'rb') as h:
        return pickle.load(h)
    
@st.cache
def unique_fos_level(df):
    return sorted(df['level'].unique())[1:]

def unique_fos(df, level, num):
    return list(df[df['level']==level].name.value_counts().index[:num])

@st.cache(allow_output_mutation=True)
def load_bert_model(name='distilbert-base-nli-stsb-mean-tokens'):
    # Instantiate the sentence-level DistilBERT
    return SentenceTransformer(name)

@st.cache
def load_faiss_index():
    with open('faiss_index.pickle', 'rb') as h:
        return pickle.load(h)

def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return [i for i in I[0]]

def main():
    data = read_data()
    fos_level = unique_fos_level(data)
    model = load_bert_model()
    faiss_index = faiss.deserialize_index(load_faiss_index())
    author_data = read_author_data()
    
    st.title("ACL Publications Explorer")
    
    filter_year = st.sidebar.slider("Filter by year", 2000, 2020, (2000, 2020), 1)
    filter_fos_level = st.sidebar.selectbox(
        "Choose Field of Study level", fos_level)
    fields_of_study = unique_fos(data, filter_fos_level, 25)
    filter_fos = st.sidebar.multiselect(
        "Choose Fields of Study", fields_of_study)
    author_input = st.sidebar.text_input("Search by author name")
    # User search
    user_input = st.sidebar.text_area("Search by paper title")
    num_results = st.sidebar.slider("Number of search results", 10, 150, 10)

    
    if filter_fos and not user_input and not author_input:
        frame = data[(data.name.isin(filter_fos)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1]))]
        color_on_fos = True
    elif filter_fos and user_input and not author_input:
        encoded_user_input = vector_search([user_input], model, faiss_index, num_results)
        frame = data[(data.name.isin(filter_fos)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1])) & (data.id.isin(encoded_user_input))]
        color_on_fos = True
    elif filter_fos and user_input and author_input:
        ids = author_data[author_data.name==author_input]['paper_id']
        encoded_user_input = vector_search([user_input], model, faiss_index, num_results)
        frame = data[(data.name.isin(filter_fos)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1])) & (data.id.isin(encoded_user_input)) & (data.id.isin(ids))]
        color_on_fos = True
    elif filter_fos and not user_input and author_input:
        ids = author_data[author_data.name==author_input]['paper_id']
        frame = data[(data.name.isin(filter_fos)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1])) & (data.id.isin(ids))]
        color_on_fos = True
    elif not filter_fos and user_input and not author_input:
        encoded_user_input = vector_search([user_input], model, faiss_index, num_results)
        frame = data[data.id.isin(encoded_user_input) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1]))]
        color_on_fos = False
    elif not filter_fos and user_input and author_input:
        encoded_user_input = vector_search([user_input], model, faiss_index, num_results=150)
        ids = author_data[author_data.name==author_input]['paper_id']
        frame = data[(data.id.isin(ids)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1])) & (data.id.isin(encoded_user_input))]
        color_on_fos = False
    elif not filter_fos and not user_input and author_input:
        ids = author_data[author_data.name==author_input]['paper_id']
        frame = data[(data.id.isin(ids)) & (data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1]))]
        color_on_fos = False
    else:
        frame = data[(data.year>=str(filter_year[0])) & (data.year<=str(filter_year[1]))]
        color_on_fos = False

    if color_on_fos:
        chart = alt.Chart(frame.drop_duplicates('id')).mark_point().encode(
            alt.X('Component 1', scale=alt.Scale(domain=(1, 16))),
            alt.Y('Component 2', scale=alt.Scale(domain=(0, 18))),
            alt.Color('name', title='Field of Study'),
            alt.Size('citations', scale=alt.Scale(range=[10,500]), title='Citations'),
            href='source:N',
            tooltip=['title', 'year']
        ).interactive().properties(width=650, height=500)

    else:
        chart = alt.Chart(frame.drop_duplicates('id')).mark_point().encode(
            alt.X('Component 1', scale=alt.Scale(domain=(1, 16))),
            alt.Y('Component 2', scale=alt.Scale(domain=(0, 18))),
            alt.Size('citations', scale=alt.Scale(range=[10,500]), title='Citations'),
            href='source:N',
            tooltip=['title', 'year']
        ).interactive().properties(width=650, height=500)

    bar_data = pd.DataFrame(frame[frame.level==filter_fos_level].groupby('name')['id'].count()).reset_index().sort_values('id', ascending=False)[:30]
    barchart = alt.Chart(bar_data).mark_bar().encode(alt.X('name', sort='-y', title='Fields of Study'), alt.Y('id', title='Count')).properties(width=650, height=150)
    c = (chart & barchart)
    st.altair_chart(c, use_container_width=True)

    st.subheader("How to use this app")
    st.write(f"""
    This application is intended for the visual exploration and discovery of research publications that have been presented at the ACL (Annual Meeting of the Association for Computational Linguistics).

Every particle in the scatterplot is an academic publication. The particles are positioned in space based on the semantic similarity of the paper titles; the closer two points are, the more semantically similar their titles. You can hover over the particles to read their titles and you can click them to be redirected to the original source. You can zoom in the visualisation by scrolling and you can reset the view by double clicking the white space within the figure. Regarding the bar chart, it shows the most used Fields of Study for the papers shown in the scatterplot.

You can also **search** for publications by paper titles (more information below). 

#### Filters
You can refine your query based on the publication year, paper content, field of study and author. You can also combine any of the filter for more granular searches.
- **Filter by year**: Select a time range for the papers. For example, drag both sliders to 2020 to find out the papers that will be presented at ACL 2020.
- **Field of Study level**: Microsoft Academic Graph uses a 6-level hierarchy where level 0 contains high level disciplines such as Computer science and level 5 contains the most granular paper keywords. This filter will change what's shown in the bar chart as well as the available options in the filter below.
- ** Fields of Study**: Select the Fields of Study to be displayed in the visualisations. The available options are affected by your selection in the above filter.
- **Search by author name**: Find an author's publications. **Note**: You need to type in the exact name.
- **Search by paper title**: Type in a paper title and find its most relevant relevant publications. You should use at least a sentence to receive meaningful results.
- **Number of search results**: Specify the number of papers to be returned when you search by paper title.
    """)

    st.subheader("About")
    st.write(
            f"""
I am [Kostas](http://kstathou.github.io/) and I work at the intersection of knowledge discovery, data engineering and scientometrics. I am a Mozilla Open Science Fellow and a Principal Data Science Researcher at Nesta. I am currently working on [Orion](https://orion-search.org/) (work in progress), an open-source knowledge discovery and research measurement tool. 

If you have any questions or would like to learn more about it, you can find me on [twitter](https://twitter.com/kstathou) or send me an email at kostas@mozillafoundation.org
    """
        )

    st.subheader("Appendix: Data & methods")
    st.write(f"""
    I collected all of the publications from [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/project/academic-knowledge/) that were published between 2000 and 2020 and were presented at the ACL.

I fetched 8,724 publications. To create the 2D visualisation, I encoded the paper titles to dense vectors using a [sentence-DistilBERT](https://github.com/UKPLab/sentence-transformers) model. That produced a 768-dimensional vector for each paper which I projected to a 2D space with [UMAP](https://umap-learn.readthedocs.io/en/latest/). For the paper title search engine, I indexed the vectors with [Faiss](https://github.com/facebookresearch/faiss/tree/master/python).
""")

if __name__ == '__main__':
    main()
