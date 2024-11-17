import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from joblib import load
import logging


class RecommendationSystem:
    def __init__(self, data_path, model_path):
        self.data = pd.read_csv(data_path)
        self.data['Customer ID'] = self.data['Customer ID'].astype(str)  # Ensure 'Customer ID' is a string
        self.model = load(model_path)
        logging.info("Model loaded successfully with joblib!")

        # Data Preprocessing
        self.data['Age Group'] = pd.cut(self.data['Age'], bins=[0, 25, 35, 50, 100], labels=['<25', '25-35', '35-50', '>50'])
        self.data['weighted_liked'] = self.data['Review Rating'].apply(lambda x: 3.0 if x >= 4.0 else 0.5)

        # User and item features
        user_features = [f"Age Group:{group}" for group in self.data['Age Group'].unique()] + \
                        [f"Gender:{gender}" for gender in self.data['Gender'].unique()] + \
                        [f"Previous Purchases:{purchases}" for purchases in self.data['Previous Purchases'].unique()]

        item_features = list(self.data['Category'].unique()) + \
                        list(self.data['Season'].unique()) + \
                        list(self.data['Color'].unique())

        # Dataset
        self.dataset = Dataset()
        self.dataset.fit(
            self.data['Customer ID'],
            self.data['Item Purchased'],
            user_features=user_features,
            item_features=item_features
        )

        # Interactions and feature matrices
        self.interactions_matrix, _ = self.dataset.build_interactions(
            [(x['Customer ID'], x['Item Purchased'], x['weighted_liked']) for _, x in self.data.iterrows()]
        )
        self.user_features_matrix = self.dataset.build_user_features(
            [(x['Customer ID'], [f"Age Group:{x['Age Group']}", f"Gender:{x['Gender']}",
                                 f"Previous Purchases:{x['Previous Purchases']}"])
             for _, x in self.data.iterrows()]
        )
        self.item_features_matrix = self.dataset.build_item_features(
            [(x['Item Purchased'], [x['Category'], x['Season'], x['Color']])
             for _, x in self.data.iterrows()]
        )

    def get_fallback_recommendations(self, top_n=10):
        # Recommend the most popular items by purchase count
        popular_items = (
            self.data.groupby('Item Purchased')
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        fallback_recommendations = [
            {
                "Item": item,
                "Category": self.data[self.data['Item Purchased'] == item]['Category'].values[0]
            }
            for item in popular_items
        ]
        return fallback_recommendations

    def recommend_similar_items_with_scores(self, clicked_product, top_n=10):
        item_index = self.dataset.mapping()[2].get(clicked_product)

        if item_index is None:
            logging.error(f"Clicked product '{clicked_product}' not found in mappings.")
            return []

        # Predict scores for all items based on similarity to clicked_product
        _, item_embeddings = self.model.get_item_representations(self.item_features_matrix)
        similarities = np.dot(item_embeddings, item_embeddings[item_index])

        item_mapping = {v: k for k, v in self.dataset.mapping()[2].items()}
        scored_items = [
            (item_mapping[i], float(similarities[i]))  # Explicitly convert to Python float
            for i in np.argsort(-similarities) if item_mapping[i] != clicked_product
        ][:top_n]

        recommendations = []
        for item, similarity in scored_items:
            category = self.data[self.data['Item Purchased'] == item]['Category'].values[0]
            recommendations.append({
                "Item": item,
                "Category": category,
                "Similarity": similarity  # Already converted to Python float
            })

        if not recommendations:
            logging.warning(f"No similar items found for product '{clicked_product}'.")
        return recommendations

    def combined_recommendations(self, clicked_product, top_n=10):
        # Get recommendations based on clicked_product
        item_recommendations = self.recommend_similar_items_with_scores(clicked_product, top_n=top_n)

        # If no recommendations, return fallback recommendations
        if not item_recommendations:
            return self.get_fallback_recommendations(top_n=top_n)

        return item_recommendations
