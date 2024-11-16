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
    
    def recommend_similar_items_with_scores(self, clicked_product, customer_id, top_n=10):
        item_index = self.dataset.mapping()[2].get(clicked_product)
        user_index = self.dataset.mapping()[0].get(customer_id)
        
        if item_index is None or user_index is None:
            logging.error(f"Clicked product '{clicked_product}' or customer '{customer_id}' not found in mappings.")
            return []

        item_ids = np.arange(len(self.dataset.mapping()[2]))
        scores = self.model.predict(
            user_ids=user_index,
            item_ids=item_ids,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        )
        
        _, item_embeddings = self.model.get_item_representations(self.item_features_matrix)
        similarities = np.dot(item_embeddings, item_embeddings[item_index])

        item_mapping = {v: k for k, v in self.dataset.mapping()[2].items()}
        scored_items = [
            (item_mapping[i], similarities[i], scores[i]) 
            for i in np.argsort(-similarities) if item_mapping[i] != clicked_product
        ][:top_n]

        # Log the structure of scored_items to inspect
        logging.debug(f"Scored items: {scored_items}")

        recommendations = []
        for item_data in scored_items:
            # Log the type and length of item_data
            logging.debug(f"Item data type: {type(item_data)}, Length: {len(item_data)}, Value: {item_data}")
            
            # Safely unpack only if there are exactly 3 values
            if len(item_data) == 3:  # Ensure it contains exactly 3 values
                item, similarity, score = item_data
                category = self.data[self.data['Item Purchased'] == item]['Category'].values[0]
                recommendations.append({
                    "Item": item,
                    "Category": category,
                    "Similarity": similarity,
                    "Score": score
                })
            else:
                logging.error(f"Unexpected structure in scored_items: {item_data}")
        
        if not recommendations:
            logging.warning(f"No similar items found for product '{clicked_product}'.")
        return recommendations

    def recommend_similar_users_with_scores(self, customer_id, top_n=10):
        # Check if the customer exists in the dataset mappings
        user_mapping = self.dataset.mapping()[0]
        if customer_id not in user_mapping:
            logging.error(f"User '{customer_id}' not found in user mapping.")
            return []

        user_index = user_mapping[customer_id]
        
        # Compute user similarities
        _, user_embeddings = self.model.get_user_representations(self.user_features_matrix)
        similarities = np.dot(user_embeddings, user_embeddings[user_index])
        
        user_mapping = {v: k for k, v in self.dataset.mapping()[0].items()}
        top_users = [
            user_mapping[i] for i in np.argsort(-similarities) if user_mapping[i] != customer_id
        ][:top_n]
        
        # Predict item scores for top similar users
        item_ids = np.arange(len(self.dataset.mapping()[2]))
        recommendations = []
        for similar_user in top_users:
            similar_user_index = self.dataset.mapping()[0][similar_user]
            similar_scores = self.model.predict(
                user_ids=similar_user_index,
                item_ids=item_ids,
                user_features=self.user_features_matrix,
                item_features=self.item_features_matrix
            )
            item_mapping = {v: k for k, v in self.dataset.mapping()[2].items()}
            recommended_items = [
                (item_mapping[i], score) for i, score in zip(np.argsort(-similar_scores), similar_scores)
            ][:top_n]
            recommendations.extend(recommended_items)
        
        return recommendations[:top_n]

    def combined_recommendations(self, clicked_product, customer_id, top_n=10):
        # Lấy gợi ý từ sản phẩm
        item_recommendations = self.recommend_similar_items_with_scores(clicked_product, customer_id, top_n=top_n)
        
        # Chuyển item_recommendations sang định dạng {item: category}
        item_recommend_dict = {
            item_data["Item"]: item_data["Category"] for item_data in item_recommendations
        }
        
        # Lấy gợi ý từ người dùng
        user_recommendations = self.recommend_similar_users_with_scores(customer_id, top_n=top_n)
        
        # Kết hợp gợi ý từ người dùng vào dict
        for item, _ in user_recommendations:
            if item not in item_recommend_dict:
                category = self.data[self.data['Item Purchased'] == item]['Category'].values[0]
                item_recommend_dict[item] = category

        # Lọc các sản phẩm duy nhất và trả về danh sách
        unique_items = set()
        final_recommendations = []

        for item, category in item_recommend_dict.items():
            if item not in unique_items:
                unique_items.add(item)
                final_recommendations.append({
                    "Item": item,
                    "Category": category
                })
                if len(final_recommendations) >= top_n:
                    break

        return final_recommendations
