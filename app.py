from flask import Flask, request, jsonify
import logging
from recommendation_system import RecommendationSystem

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize recommendation system (ensure paths are correct)
try:
    recommender = RecommendationSystem('shopping_behavior_updated.csv', 'best_lightfm_model.joblib')
except Exception as e:
    logging.error(f"Failed to initialize Recommendation System: {e}")
    recommender = None

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse JSON request
        content = request.json
        logging.info(f"Received request payload: {content}")

        clicked_product = content.get('clicked_product')

        # Validate input
        if not clicked_product:
            logging.warning("Invalid request: Missing 'clicked_product'.")
            return jsonify({"error": "Please provide 'clicked_product'."}), 400

        # Check if recommender system is initialized
        if recommender is None:
            logging.error("Recommendation System is not initialized.")
            return jsonify({"error": "Recommendation System is currently unavailable."}), 503

        logging.info(f"Generating recommendations for clicked_product: {clicked_product}")

        # Generate recommendations
        recommendations = recommender.combined_recommendations(clicked_product, top_n=10)

        if not recommendations:
            logging.warning("No recommendations found.")
            return jsonify({
                "message": "No recommendations available. Showing popular items.",
                "recommendations": recommender.get_fallback_recommendations(top_n=10)
            }), 404

        logging.info(f"Recommendations generated successfully for clicked_product={clicked_product}.")
        return jsonify(recommendations)

    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        return jsonify({"error": "An error occurred while processing the recommendation."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
