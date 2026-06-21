import os
import joblib
import streamlit as st

st.set_page_config(page_title="SHOPPER SPECTRUM", page_icon="🛒")

# 1. Load the similarity matrix
@st.cache_resource
def load_similarity_matrix():
    # Get the current folder (pages) and go up one level to the main root folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    # Safely construct the exact path to the similarity matrix
    matrix_path = os.path.join(parent_dir, 'item_similarity_df.pkl')
    
    try:
        similarity_df = joblib.load(matrix_path)
        return similarity_df
    except FileNotFoundError:
        st.error(f"Similarity matrix not found! Looked in:\n{matrix_path}")
        return None

item_similarity_df = load_similarity_matrix()

st.title("PRODUCT RECOMMENDER")
st.write("ITEM RECOMMENDATION AND SHOPPING PREFERENCES APPLICATION")

st.warning("ENTER VALID PRODUCT NAME ONLY!")

# 2. Create the search input
search_input = st.text_input("SEARCH YOUR ITEM HERE")

# 3. Recommendation Logic
if st.button("SUBMIT"):
    if item_similarity_df is not None:
        if search_input:
            # Convert input to uppercase (dataset product names are typically uppercase)
            product_name = search_input.upper()
            
            if product_name in item_similarity_df.columns:
                st.write(f"### Recommended Products for: {product_name}")
                
                # Get the top 5 similar products (Index 1 to 6 to skip the item matching itself at index 0)
                recommended_items = item_similarity_df[product_name].sort_values(ascending=False)[1:6]
                
                # Display them as a list
                for item in recommended_items.index:
                    st.write(f"- {item}")
            else:
                st.error(f"'{search_input}' not found in the database. Please try another product.")
        else:
            st.warning("Please type a product name before hitting submit.")