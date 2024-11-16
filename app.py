from flask import Flask, request, jsonify
import logging
from recommendation_system import RecommendationSystem

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize recommendation system (ensure paths are correct)
recommender = RecommendationSystem('shopping_behavior_updated.csv', 'best_lightfm_model.joblib')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse JSON request
        content = request.json
        logging.info(f"Received request payload: {content}")

        clicked_product = content.get('clicked_product')
        customer_id = content.get('customer_id')

        # Validate inputs
        if not clicked_product or not customer_id:
            logging.warning("Invalid request: Missing 'clicked_product' or 'customer_id'.")
            return jsonify({"error": "Please provide both 'clicked_product' and 'customer_id'."}), 400

        # Check if recommender system is initialized
        if recommender is None:
            logging.error("Recommendation System is not initialized.")
            return jsonify({"error": "Recommendation System is currently unavailable."}), 503

        logging.info(f"Generating recommendations for clicked_product: {clicked_product} and customer_id: {customer_id}")

        # Generate recommendations
        recommendations = recommender.combined_recommendations(
            clicked_product, customer_id, top_n=10
        )

        if not recommendations:
            logging.warning("No recommendations found.")
            return jsonify({"error": "No recommendations found."}), 404

        logging.info(f"Recommendations generated successfully for customer_id={customer_id}.")
        return jsonify(recommendations)  # Return only Item and Category as JSON

    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        return jsonify({"error": "An error occurred while processing the recommendation."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

