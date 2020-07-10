# ACL explorer #
[vid](https://youtu.be/WuBZu8KeMFw)

This application is intended for the visual exploration and discovery of research publications that have been presented at the ACL (Annual Meeting of the Association for Computational Linguistics).

Every particle in the scatterplot is an academic publication. The particles are positioned in space based on the semantic similarity of the paper titles; the closer two points are, the more semantically similar their titles. You can hover over the particles to read their titles and you can click them to be redirected to the original source. You can zoom in the visualisation by scrolling and you can reset the view by double clicking the white space within the figure. Regarding the bar chart, it shows the most used Fields of Study for the papers shown in the scatterplot.

You can also search for publications by paper titles (more information below).

## Filters ##

You can refine your query based on the publication year, paper content, field of study and author. You can also combine any of the filter for more granular searches.

- Filter by year: Select a time range for the papers. For example, drag both sliders to 2020 to find out the papers that will be presented at ACL 2020.
- Field of Study level: Microsoft Academic Graph uses a 6-level hierarchy where level 0 contains high level disciplines such as Computer science and level 5 contains the most granular paper keywords. This filter will change what's shown in the bar chart as well as the available options in the filter below.
- Fields of Study: Select the Fields of Study to be displayed in the visualisations. The available options are affected by your selection in the above filter.
- Search by author name: Find an author's publications. Note: You need to type in the exact name.
- Search by paper title: Type in a paper title and find its most relevant publications. You should use at least a sentence to receive meaningful results.
- Number of search results: Specify the number of papers to be returned when you search by paper title.

## About ##

I am Kostas and I work at the intersection of knowledge discovery, data engineering and scientometrics. I am a Mozilla Open Science Fellow and a Principal Data Science Researcher at Nesta. I am currently working on [Orion](https://www.orion-search.org/), an open-source knowledge discovery and research measurement tool.

If you have any questions or would like to learn more about it, you can find me on [twitter](https://twitter.com/kstathou) or send me an email at kostas@mozillafoundation.org

## Appendix: Data & methods ##

I collected all of the publications from Microsoft Academic Graph that were published between 2000 and 2020 and were presented at the ACL.

I fetched 8,724 publications. To create the 2D visualisation, I encoded the paper titles to dense vectors using a sentence-DistilBERT model. That produced a 768-dimensional vector for each paper which I projected to a 2D space with UMAP. For the paper title search engine, I indexed the vectors with Faiss.
