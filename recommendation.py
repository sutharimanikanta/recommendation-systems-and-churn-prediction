import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def prepare_recommendation_system(df):
    """Initialize and prepare the recommendation system."""
    try:
        interaction_data = df[["CustomerID", "ServiceUsage1", "ServiceUsage2", "ServiceUsage3", 
                               "TotalCharges", "MonthlyCharges", "Tenure"]].copy()
        interaction_data.fillna(0, inplace=True)
        interaction_matrix = interaction_data.set_index("CustomerID")

        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(interaction_matrix)
        normalized_interaction_matrix = pd.DataFrame(
            normalized_matrix, 
            columns=interaction_matrix.columns, 
            index=interaction_matrix.index
        )

        sparse_matrix = csr_matrix(normalized_interaction_matrix)
        model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
        model.fit(sparse_matrix)

        return {
            'interaction_matrix': interaction_matrix,
            'normalized_matrix': normalized_interaction_matrix,
            'sparse_matrix': sparse_matrix,
            'model': model,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_recommendations(customer_id, rec_system, df, k=5):
    """Get personalized recommendations for a customer."""
    try:
        if customer_id not in rec_system['interaction_matrix'].index:
            return {
                "error": "Customer ID not found",
                "sample_ids": list(rec_system['interaction_matrix'].index[:5])
            }

        customer_index = rec_system['normalized_matrix'].index.get_loc(customer_id)
        distances, indices = rec_system['model'].kneighbors(
            rec_system['sparse_matrix'][customer_index].toarray(), 
            n_neighbors=k+1
        )

        similar_customers = rec_system['normalized_matrix'].index[indices.flatten()].tolist()
        similar_customers.remove(customer_id)

        target_customer_data = rec_system['interaction_matrix'].loc[customer_id]
        recommendations = {}
        for similar_customer in similar_customers:
            similar_customer_data = rec_system['interaction_matrix'].loc[similar_customer]
            similarity_score = 1 / (1 + distances[0][similar_customers.index(similar_customer) + 1])
            
            for service in ["ServiceUsage1", "ServiceUsage2", "ServiceUsage3"]:
                if target_customer_data[service] < similar_customer_data[service]:
                    weight = 1.0
                    if target_customer_data['Tenure'] > 24:
                        weight *= 1.2
                    if target_customer_data['MonthlyCharges'] > df['MonthlyCharges'].median():
                        weight *= 1.1
                    if service == "ServiceUsage2":
                        weight += 0.2
                    if service == "ServiceUsage1":
                        weight -= 0.1

                    weight *= similarity_score
                    current_usage = target_customer_data[service]
                    recommended_usage = similar_customer_data[service]
                    usage_difference = recommended_usage - current_usage

                    if service not in recommendations:
                        recommendations[service] = {
                            'current_usage': float(current_usage),
                            'recommended_usage': float(recommended_usage),
                            'weighted_score': 0,
                            'potential_increase': float(usage_difference),
                            'similar_customers_count': 0
                        }

                    recommendations[service]['weighted_score'] += weight
                    recommendations[service]['similar_customers_count'] += 1

        formatted_recommendations = []
        for service, details in recommendations.items():
            if details['similar_customers_count'] > 0:
                formatted_recommendations.append({
                    'service': service,
                    'current_usage': details['current_usage'],
                    'recommended_usage': details['recommended_usage'],
                    'potential_increase': details['potential_increase'],
                    'confidence_score': details['weighted_score'] / details['similar_customers_count'],
                    'supporting_customers': details['similar_customers_count'],
                    'message': f"Based on {details['similar_customers_count']} similar customers, "
                             f"we recommend increasing your {service} usage by "
                             f"{(details['potential_increase']):.1f} units."
                })

        formatted_recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        return {
            'customer_id': customer_id,
            'recommendations': formatted_recommendations,
        }
    except Exception as e:
        return {"error": f"Error generating recommendations: {str(e)}"}
