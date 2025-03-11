"""
K-Nearest Neighbors and Recommendation Systems - Chapter 10 Implementation
=======================================================================

This module demonstrates the concepts from Chapter 10 of "Grokking Algorithms":
1. K-Nearest Neighbors (KNN) algorithm
2. Building recommendation systems
3. Different distance metrics
4. Normalization of ratings
5. Handling influencers and weighted recommendations
6. Determining optimal K value

"""

import logging
import time
import math
import random
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- User and Movie Rating System --------------------

class User:
    """Represents a user with movie ratings."""
    
    def __init__(self, user_id: str, is_influencer: bool = False, influence_weight: float = 1.0):
        """
        Initialize a user with an ID and movie ratings.
        
        Args:
            user_id: Unique identifier for the user
            is_influencer: Whether this user is an influencer
            influence_weight: Weight of this user's ratings if they're an influencer
        """
        self.user_id = user_id
        self.ratings = {}  # Movie -> Rating
        self.is_influencer = is_influencer
        self.influence_weight = influence_weight if is_influencer else 1.0
    
    def rate_movie(self, movie: str, rating: float) -> None:
        """
        Rate a movie.
        
        Args:
            movie: Movie title
            rating: Rating value (typically 1-5)
        """
        self.ratings[movie] = rating
    
    def get_rating(self, movie: str) -> Optional[float]:
        """
        Get the user's rating for a movie.
        
        Args:
            movie: Movie title
            
        Returns:
            Rating if the user has rated the movie, None otherwise
        """
        return self.ratings.get(movie)
    
    def get_all_rated_movies(self) -> Set[str]:
        """
        Get all movies rated by the user.
        
        Returns:
            Set of movie titles
        """
        return set(self.ratings.keys())
    
    def get_average_rating(self) -> float:
        """
        Calculate the average rating given by the user.
        
        Returns:
            Average rating or 0 if no ratings
        """
        if not self.ratings:
            return 0
        return sum(self.ratings.values()) / len(self.ratings)
    
    def get_normalized_ratings(self, target_average: float = 3.0) -> Dict[str, float]:
        """
        Get normalized ratings to account for different rating scales.
        
        Args:
            target_average: Target average rating
            
        Returns:
            Dictionary of normalized ratings
        """
        if not self.ratings:
            return {}
        
        current_avg = self.get_average_rating()
        if current_avg == 0:
            return {}
        
        # Simple shift normalization
        shift = target_average - current_avg
        return {movie: min(5, max(1, rating + shift)) for movie, rating in self.ratings.items()}
    
    def __str__(self) -> str:
        """String representation of the user."""
        status = "Influencer" if self.is_influencer else "Regular user"
        avg = self.get_average_rating()
        return f"{self.user_id} ({status}): {len(self.ratings)} ratings, avg={avg:.2f}"

class MovieRecommender:
    """K-Nearest Neighbors based movie recommender system."""
    
    def __init__(self, normalize_ratings: bool = False, 
                use_influencers: bool = False,
                distance_metric: str = "euclidean"):
        """
        Initialize the recommender system.
        
        Args:
            normalize_ratings: Whether to normalize ratings to account for different scales
            use_influencers: Whether to give more weight to influencers' ratings
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')
        """
        self.users = {}  # User ID -> User object
        self.movies = set()  # Set of all movies
        self.normalize_ratings = normalize_ratings
        self.use_influencers = use_influencers
        self.distance_metric = distance_metric
        
        # Set the distance function based on the metric
        if distance_metric == "manhattan":
            self.distance_func = self._manhattan_distance
        elif distance_metric == "cosine":
            self.distance_func = self._cosine_similarity
        else:
            self.distance_func = self._euclidean_distance
    
    def add_user(self, user: User) -> None:
        """
        Add a user to the system.
        
        Args:
            user: User object
        """
        self.users[user.user_id] = user
        self.movies.update(user.get_all_rated_movies())
    
    def _get_common_movies(self, user1: User, user2: User) -> List[str]:
        """
        Get movies rated by both users.
        
        Args:
            user1: First user
            user2: Second user
            
        Returns:
            List of movies rated by both users
        """
        return list(user1.get_all_rated_movies() & user2.get_all_rated_movies())
    
    def _euclidean_distance(self, user1: User, user2: User) -> float:
        """
        Calculate Euclidean distance between two users based on ratings.
        
        Args:
            user1: First user
            user2: Second user
            
        Returns:
            Euclidean distance (smaller means more similar)
        """
        common_movies = self._get_common_movies(user1, user2)
        
        if not common_movies:
            return float('inf')  # No common movies, very dissimilar
        
        # Get ratings (normalized if enabled)
        ratings1 = user1.get_normalized_ratings() if self.normalize_ratings else user1.ratings
        ratings2 = user2.get_normalized_ratings() if self.normalize_ratings else user2.ratings
        
        # Calculate squared differences
        sum_squared_diff = sum((ratings1[movie] - ratings2[movie]) ** 2 for movie in common_movies)
        
        return math.sqrt(sum_squared_diff)
    
    def _manhattan_distance(self, user1: User, user2: User) -> float:
        """
        Calculate Manhattan distance between two users based on ratings.
        
        Args:
            user1: First user
            user2: Second user
            
        Returns:
            Manhattan distance (smaller means more similar)
        """
        common_movies = self._get_common_movies(user1, user2)
        
        if not common_movies:
            return float('inf')  # No common movies, very dissimilar
        
        # Get ratings (normalized if enabled)
        ratings1 = user1.get_normalized_ratings() if self.normalize_ratings else user1.ratings
        ratings2 = user2.get_normalized_ratings() if self.normalize_ratings else user2.ratings
        
        # Calculate absolute differences
        sum_abs_diff = sum(abs(ratings1[movie] - ratings2[movie]) for movie in common_movies)
        
        return sum_abs_diff
    
    def _cosine_similarity(self, user1: User, user2: User) -> float:
        """
        Calculate cosine similarity between two users based on ratings.
        
        Args:
            user1: First user
            user2: Second user
            
        Returns:
            Cosine similarity (higher means more similar, converted to distance)
        """
        common_movies = self._get_common_movies(user1, user2)
        
        if not common_movies:
            return float('inf')  # No common movies, very dissimilar
        
        # Get ratings (normalized if enabled)
        ratings1 = user1.get_normalized_ratings() if self.normalize_ratings else user1.ratings
        ratings2 = user2.get_normalized_ratings() if self.normalize_ratings else user2.ratings
        
        # Calculate dot product
        dot_product = sum(ratings1[movie] * ratings2[movie] for movie in common_movies)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(ratings1[movie] ** 2 for movie in common_movies))
        magnitude2 = math.sqrt(sum(ratings2[movie] ** 2 for movie in common_movies))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return float('inf')
        
        # Cosine similarity is between -1 and 1, with 1 being most similar
        # Convert to distance (0 to 2, with 0 being most similar)
        similarity = dot_product / (magnitude1 * magnitude2)
        distance = 1 - similarity  # Convert to distance (0 to 2)
        
        return distance
    
    def find_nearest_neighbors(self, user_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a user.
        
        Args:
            user_id: ID of the user
            k: Number of neighbors to find
            
        Returns:
            List of (user_id, distance) tuples for the k nearest neighbors
        """
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return []
        
        user = self.users[user_id]
        
        # Calculate distances to all other users
        distances = []
        for other_id, other_user in self.users.items():
            if other_id != user_id:
                distance = self.distance_func(user, other_user)
                distances.append((other_id, distance))
        
        # Sort by distance and return k nearest
        return sorted(distances, key=lambda x: x[1])[:k]
    
    def recommend_movies(self, user_id: str, k: int = 5, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend movies for a user based on k nearest neighbors.
        
        Args:
            user_id: ID of the user
            k: Number of neighbors to consider
            top_n: Number of top recommendations to return
            
        Returns:
            List of (movie, predicted_rating) tuples
        """
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return []
        
        user = self.users[user_id]
        user_movies = user.get_all_rated_movies()
        
        # Find nearest neighbors
        neighbors = self.find_nearest_neighbors(user_id, k)
        
        # Calculate predicted ratings for unrated movies
        predicted_ratings = {}
        
        for movie in self.movies:
            if movie in user_movies:
                continue  # Skip already rated movies
            
            # Weighted sum of ratings from neighbors
            weighted_sum = 0
            weight_sum = 0
            
            for neighbor_id, distance in neighbors:
                neighbor = self.users[neighbor_id]
                rating = neighbor.get_rating(movie)
                
                if rating is not None:
                    # Convert distance to weight (closer = higher weight)
                    # Add a small constant to avoid division by zero
                    weight = 1 / (distance + 0.1)
                    
                    # Apply influencer weight if enabled
                    if self.use_influencers:
                        weight *= neighbor.influence_weight
                    
                    weighted_sum += rating * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                predicted_ratings[movie] = weighted_sum / weight_sum
        
        # Sort by predicted rating and return top N
        sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_ratings[:top_n]
    
    def calculate_optimal_k(self, validation_ratio: float = 0.2, 
                          max_k: int = 20) -> Tuple[int, float]:
        """
        Calculate the optimal k value using cross-validation.
        
        Args:
            validation_ratio: Ratio of ratings to use for validation
            max_k: Maximum k value to consider
            
        Returns:
            Tuple of (optimal_k, error)
        """
        # Collect all ratings
        all_ratings = []
        for user_id, user in self.users.items():
            for movie, rating in user.ratings.items():
                all_ratings.append((user_id, movie, rating))
        
        # Shuffle ratings
        random.shuffle(all_ratings)
        
        # Split into training and validation sets
        split_idx = int(len(all_ratings) * (1 - validation_ratio))
        training_ratings = all_ratings[:split_idx]
        validation_ratings = all_ratings[split_idx:]
        
        # Create a temporary recommender with the training set
        temp_recommender = MovieRecommender(
            normalize_ratings=self.normalize_ratings,
            use_influencers=self.use_influencers,
            distance_metric=self.distance_metric
        )
        
        # Add users and their ratings
        for user_id, movie, rating in training_ratings:
            if user_id not in temp_recommender.users:
                # Copy user properties from original recommender
                original_user = self.users[user_id]
                new_user = User(
                    user_id=user_id,
                    is_influencer=original_user.is_influencer,
                    influence_weight=original_user.influence_weight
                )
                temp_recommender.add_user(new_user)
            
            temp_recommender.users[user_id].rate_movie(movie, rating)
            temp_recommender.movies.add(movie)
        
        # Calculate error for different k values
        k_values = range(1, max_k + 1)
        errors = []
        
        for k in k_values:
            total_error = 0
            count = 0
            
            for user_id, movie, actual_rating in validation_ratings:
                if user_id not in temp_recommender.users:
                    continue
                
                # Temporarily remove this rating if it exists
                user = temp_recommender.users[user_id]
                original_rating = user.get_rating(movie)
                if original_rating is not None:
                    user.ratings.pop(movie)
                
                # Get recommendations with current k
                recommendations = temp_recommender.recommend_movies(user_id, k=k)
                
                # Find if the movie is in recommendations
                predicted_rating = None
                for rec_movie, rec_rating in recommendations:
                    if rec_movie == movie:
                        predicted_rating = rec_rating
                        break
                
                # If not in top recommendations, try direct prediction
                if predicted_rating is None:
                    neighbors = temp_recommender.find_nearest_neighbors(user_id, k=k)
                    
                    if neighbors:
                        # Direct prediction calculation
                        weighted_sum = 0
                        weight_sum = 0
                        
                        for neighbor_id, distance in neighbors:
                            neighbor = temp_recommender.users[neighbor_id]
                            neighbor_rating = neighbor.get_rating(movie)
                            
                            if neighbor_rating is not None:
                                weight = 1 / (distance + 0.1)
                                if temp_recommender.use_influencers:
                                    weight *= neighbor.influence_weight
                                
                                weighted_sum += neighbor_rating * weight
                                weight_sum += weight
                        
                        if weight_sum > 0:
                            predicted_rating = weighted_sum / weight_sum
                
                # Calculate error if prediction exists
                if predicted_rating is not None:
                    total_error += (predicted_rating - actual_rating) ** 2
                    count += 1
                
                # Restore original rating if it existed
                if original_rating is not None:
                    user.rate_movie(movie, original_rating)
            
            # Calculate mean squared error
            mse = total_error / count if count > 0 else float('inf')
            errors.append(mse)
            logger.info(f"k={k}, MSE={mse:.4f}")
        
        # Find optimal k
        optimal_k = k_values[errors.index(min(errors))]
        min_error = min(errors)
        
        return optimal_k, min_error
    
    def calculate_sqrt_n_recommendation(self) -> int:
        """
        Calculate the recommended k value based on the square root of the number of users.
        
        Returns:
            Recommended k value
        """
        n_users = len(self.users)
        return int(math.sqrt(n_users))

# -------------------- Visualization Functions --------------------

def visualize_user_ratings(users: Dict[str, User], 
                         title: str = "User Ratings Distribution"):
    """
    Visualize the distribution of ratings for each user.
    
    Args:
        users: Dictionary of User objects
        title: Title for the visualization
    """
    # Collect ratings data
    data = []
    for user_id, user in users.items():
        for movie, rating in user.ratings.items():
            data.append({
                'User': user_id,
                'Movie': movie,
                'Rating': rating,
                'Type': 'Influencer' if user.is_influencer else 'Regular'
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot rating distribution by user
    plt.subplot(2, 2, 1)
    sns.boxplot(x='User', y='Rating', data=df, palette='Set3')
    plt.xticks(rotation=45, ha='right')
    plt.title('Rating Distribution by User')
    plt.xlabel('')
    plt.tight_layout()
    
    # Plot rating histogram
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='Rating', hue='Type', multiple='stack', discrete=True)
    plt.title('Rating Distribution')
    plt.tight_layout()
    
    # Plot heatmap of user-movie ratings
    plt.subplot(2, 1, 2)
    pivot_df = df.pivot_table(index='User', columns='Movie', values='Rating')
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5)
    plt.title('User-Movie Ratings Matrix')
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle(title, fontsize=16, y=1.02)
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Ratings visualization saved as '{filename}'")
    plt.close()

def visualize_recommendations(user_id: str, recommendations: List[Tuple[str, float]], 
                             neighbors: List[Tuple[str, User]], users: Dict[str, User],
                             title: str = "Movie Recommendations"):
    """
    Visualize movie recommendations and neighbor influence.
    
    Args:
        user_id: ID of the user
        recommendations: List of (movie, rating) tuples
        neighbors: List of (user_id, distance) tuples
        users: Dictionary of User objects
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot recommended movies and their ratings
    plt.subplot(2, 2, 1)
    movies, ratings = zip(*recommendations) if recommendations else ([], [])
    bars = plt.bar(movies, ratings, color='lightblue')
    plt.axhline(y=3.0, color='r', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 5.5)
    plt.title(f'Top Recommendations for {user_id}')
    plt.ylabel('Predicted Rating')
    
    # Plot nearest neighbors and distances
    plt.subplot(2, 2, 2)
    neighbor_ids, distances = zip(*neighbors) if neighbors else ([], [])
    plt.bar(neighbor_ids, [1/d for d in distances], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.title('Nearest Neighbors (Similarity)')
    plt.ylabel('Similarity (1/distance)')
    
    # Plot average ratings comparison
    plt.subplot(2, 2, 3)
    user_avg = users[user_id].get_average_rating()
    neighbor_avgs = [users[n_id].get_average_rating() for n_id in neighbor_ids]
    
    labels = [user_id] + list(neighbor_ids)
    avgs = [user_avg] + neighbor_avgs
    colors = ['blue'] + ['green' if users[n_id].is_influencer else 'lightgreen' 
                        for n_id in neighbor_ids]
    
    plt.bar(labels, avgs, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Rating Comparison')
    plt.ylabel('Average Rating')
    
    # Plot influence of each neighbor
    plt.subplot(2, 2, 4)
    influence = []
    for n_id, dist in neighbors:
        # Influence is proportional to 1/distance and influencer weight
        weight = 1 / (dist + 0.1)
        if users[n_id].is_influencer:
            weight *= users[n_id].influence_weight
        influence.append(weight)
    
    # Normalize influence to sum to 1
    if influence:
        influence = [i / sum(influence) for i in influence]
    
    plt.pie(influence, labels=neighbor_ids, autopct='%1.1f%%', 
           colors=plt.cm.Pastel1(range(len(neighbor_ids))))
    plt.title('Neighbor Influence on Recommendations')
    
    # Add overall title
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Recommendations visualization saved as '{filename}'")
    plt.close()

def visualize_k_selection(k_values: List[int], errors: List[float], optimal_k: int,
                        sqrt_n_k: int, title: str = "Optimal K Selection"):
    """
    Visualize the process of selecting the optimal k value.
    
    Args:
        k_values: List of k values
        errors: List of errors for each k
        optimal_k: Optimal k value based on lowest error
        sqrt_n_k: K value based on sqrt(N) rule of thumb
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot error vs k
    plt.plot(k_values, errors, 'o-', color='blue', markersize=8)
    
    # Highlight optimal k
    plt.axvline(x=optimal_k, color='green', linestyle='--', 
               label=f'Optimal k = {optimal_k}')
    
    # Highlight sqrt(n) k
    plt.axvline(x=sqrt_n_k, color='red', linestyle='--',
               label=f'sqrt(N) k = {sqrt_n_k}')
    
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"K selection visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercise 10.1 - Rating Normalization --------------------

def exercise_10_1():
    """
    Exercise 10.1: Normalize ratings to account for different rating scales.
    """
    logger.info("\nExercise 10.1: Rating Normalization")
    logger.info("-" * 60)
    
    # Create users with different rating scales
    yogi = User("Yogi")
    pinky = User("Pinky")
    
    # Add ratings - Yogi rates what he likes as 5, Pinky is more reserved
    # But they have similar taste in movies
    movies = [
        "Star Wars", "Empire Strikes Back", "Return of the Jedi",
        "The Matrix", "The Godfather", "Pulp Fiction",
        "Inception", "The Dark Knight", "Forrest Gump"
    ]
    
    # Yogi's ratings - higher average
    yogi.rate_movie("Star Wars", 5)
    yogi.rate_movie("Empire Strikes Back", 5)
    yogi.rate_movie("Return of the Jedi", 5)
    yogi.rate_movie("The Matrix", 5)
    yogi.rate_movie("The Godfather", 4)
    yogi.rate_movie("Pulp Fiction", 3)
    yogi.rate_movie("Inception", 4)
    yogi.rate_movie("The Dark Knight", 5)
    yogi.rate_movie("Forrest Gump", 2)
    
    # Pinky's ratings - lower average but similar preferences
    pinky.rate_movie("Star Wars", 4)
    pinky.rate_movie("Empire Strikes Back", 5)
    pinky.rate_movie("Return of the Jedi", 3)
    pinky.rate_movie("The Matrix", 4)
    pinky.rate_movie("The Godfather", 3)
    pinky.rate_movie("Pulp Fiction", 2)
    pinky.rate_movie("Inception", 3)
    pinky.rate_movie("The Dark Knight", 4)
    pinky.rate_movie("Forrest Gump", 1)
    
    # Calculate average ratings
    yogi_avg = yogi.get_average_rating()
    pinky_avg = pinky.get_average_rating()
    
    logger.info(f"Yogi's average rating: {yogi_avg:.2f}")
    logger.info(f"Pinky's average rating: {pinky_avg:.2f}")
    
    # Create two recommenders - one with normalization, one without
    recommender_without_norm = MovieRecommender(normalize_ratings=False)
    recommender_with_norm = MovieRecommender(normalize_ratings=True)
    
    # Add users to both recommenders
    recommender_without_norm.add_user(yogi)
    recommender_without_norm.add_user(pinky)
    
    # Create copies for the second recommender
    recommender_with_norm.add_user(User("Yogi"))
    recommender_with_norm.add_user(User("Pinky"))
    
    # Add the same ratings
    for movie, rating in yogi.ratings.items():
        recommender_with_norm.users["Yogi"].rate_movie(movie, rating)
    
    for movie, rating in pinky.ratings.items():
        recommender_with_norm.users["Pinky"].rate_movie(movie, rating)
    
    # Calculate distances with and without normalization
    distance_without_norm = recommender_without_norm.distance_func(yogi, pinky)
    
    # For normalized version, we need to manually calculate as it uses internal normalized ratings
    yogi_norm = recommender_with_norm.users["Yogi"]
    pinky_norm = recommender_with_norm.users["Pinky"]
    distance_with_norm = recommender_with_norm.distance_func(yogi_norm, pinky_norm)
    
    logger.info(f"Distance without normalization: {distance_without_norm:.4f}")
    logger.info(f"Distance with normalization: {distance_with_norm:.4f}")
    
    # Display normalized ratings for comparison
    logger.info("\nOriginal vs. Normalized Ratings:")
    logger.info(f"{'Movie':<20} {'Yogi':<5} {'Pinky':<5} {'Yogi (Norm)':<12} {'Pinky (Norm)':<12}")
    logger.info("-" * 60)
    
    yogi_normalized = yogi.get_normalized_ratings()
    pinky_normalized = pinky.get_normalized_ratings(target_average=yogi_avg)
    
    for movie in movies:
        yogi_rating = yogi.get_rating(movie)
        pinky_rating = pinky.get_rating(movie)
        yogi_norm_rating = yogi_normalized.get(movie, "-")
        pinky_norm_rating = pinky_normalized.get(movie, "-")
        
        logger.info(f"{movie:<20} {yogi_rating:<5} {pinky_rating:<5} " +
                   f"{yogi_norm_rating:<12} {pinky_norm_rating:<12}")
    
    # Visualize users and their ratings
    users_dict = {"Yogi": yogi, "Pinky": pinky}
    visualize_user_ratings(users_dict, "Exercise 10.1 - User Ratings Before Normalization")
    
    # Create normalized users for visualization
    yogi_norm_vis = User("Yogi (Normalized)")
    pinky_norm_vis = User("Pinky (Normalized)")
    
    for movie, rating in yogi_normalized.items():
        yogi_norm_vis.rate_movie(movie, rating)
    
    for movie, rating in pinky_normalized.items():
        pinky_norm_vis.rate_movie(movie, rating)
    
    users_norm_dict = {"Yogi (Normalized)": yogi_norm_vis, "Pinky (Normalized)": pinky_norm_vis}
    visualize_user_ratings(users_norm_dict, "Exercise 10.1 - User Ratings After Normalization")
    
    logger.info("\nConclusion for Exercise 10.1:")
    logger.info("Normalization accounts for different rating scales by adjusting ratings")
    logger.info("to a common average. This makes the distance between users with similar")
    logger.info("preferences but different rating tendencies smaller, which improves")
    logger.info("the quality of recommendations.")

# -------------------- Exercise 10.2 - Influencer Weighting --------------------

def exercise_10_2():
    """
    Exercise 10.2: Use influencer weighting in the recommendation system.
    """
    logger.info("\nExercise 10.2: Influencer Weighting")
    logger.info("-" * 60)
    
    # Create regular users
    joe = User("Joe")
    dave = User("Dave")
    
    # Create an influencer
    wes = User("Wes Anderson", is_influencer=True, influence_weight=3.0)
    
    # Add ratings
    joe.rate_movie("Caddyshack", 3)
    dave.rate_movie("Caddyshack", 4)
    wes.rate_movie("Caddyshack", 5)
    
    # More ratings to create a fuller profile
    joe.rate_movie("Ghostbusters", 4)
    joe.rate_movie("The Royal Tenenbaums", 2)
    joe.rate_movie("Groundhog Day", 5)
    
    dave.rate_movie("Ghostbusters", 5)
    dave.rate_movie("The Royal Tenenbaums", 3)
    dave.rate_movie("Moonrise Kingdom", 2)
    
    wes.rate_movie("The Royal Tenenbaums", 5)
    wes.rate_movie("Moonrise Kingdom", 5)
    wes.rate_movie("The Grand Budapest Hotel", 5)
    
    # Create two recommenders - one without influencers, one with
    recommender_without_inf = MovieRecommender(use_influencers=False)
    recommender_with_inf = MovieRecommender(use_influencers=True)
    
    # Add users to both recommenders
    recommender_without_inf.add_user(joe)
    recommender_without_inf.add_user(dave)
    recommender_without_inf.add_user(wes)
    
    # Create copies for the second recommender
    joe_copy = User("Joe")
    dave_copy = User("Dave")
    wes_copy = User("Wes Anderson", is_influencer=True, influence_weight=3.0)
    
    # Add the same ratings
    for movie, rating in joe.ratings.items():
        joe_copy.rate_movie(movie, rating)
    
    for movie, rating in dave.ratings.items():
        dave_copy.rate_movie(movie, rating)
    
    for movie, rating in wes.ratings.items():
        wes_copy.rate_movie(movie, rating)
    
    recommender_with_inf.add_user(joe_copy)
    recommender_with_inf.add_user(dave_copy)
    recommender_with_inf.add_user(wes_copy)
    
    # Create a test user who needs recommendations
    test_user = User("TestUser")
    test_user.rate_movie("Ghostbusters", 4)
    test_user.rate_movie("The Royal Tenenbaums", 4)
    
    test_user_copy = User("TestUser")
    test_user_copy.rate_movie("Ghostbusters", 4)
    test_user_copy.rate_movie("The Royal Tenenbaums", 4)
    
    recommender_without_inf.add_user(test_user)
    recommender_with_inf.add_user(test_user_copy)
    
    # Get recommendations with and without influencer weighting
    neighbors_without_inf = recommender_without_inf.find_nearest_neighbors("TestUser", k=3)
    neighbors_with_inf = recommender_with_inf.find_nearest_neighbors("TestUser", k=3)
    
    recs_without_inf = recommender_without_inf.recommend_movies("TestUser", k=3)
    recs_with_inf = recommender_with_inf.recommend_movies("TestUser", k=3)
    
    # Log results
    logger.info("Without influencer weighting:")
    logger.info(f"Nearest neighbors: {neighbors_without_inf}")
    logger.info(f"Recommendations: {recs_without_inf}")
    
    logger.info("\nWith influencer weighting:")
    logger.info(f"Nearest neighbors: {neighbors_with_inf}")
    logger.info(f"Recommendations: {recs_with_inf}")
    
    # Manually calculate the average rating for Caddyshack with and without weighting
    # For the example in the exercise
    logger.info("\nManual calculation for Caddyshack ratings:")
    
    # Without influencer weighting
    avg_without_inf = (3 + 4 + 5) / 3
    logger.info(f"Average without influencer weighting: {avg_without_inf}")
    
    # With influencer weighting (Wes Anderson counts 3 times)
    avg_with_inf = (3 + 4 + 5 + 5 + 5) / 5
    logger.info(f"Average with influencer weighting: {avg_with_inf}")
    
    # Visualize the recommendations
    users_dict = {"Joe": joe, "Dave": dave, "Wes Anderson": wes, "TestUser": test_user}
    visualize_user_ratings(users_dict, "Exercise 10.2 - User Ratings")
    
    visualize_recommendations(
        "TestUser", 
        recs_without_inf, 
        [(id, recommender_without_inf.distance_func(test_user, recommender_without_inf.users[id])) 
         for id, _ in neighbors_without_inf],
        users_dict,
        "Exercise 10.2 - Recommendations Without Influencer Weighting"
    )
    
    visualize_recommendations(
        "TestUser", 
        recs_with_inf, 
        [(id, recommender_with_inf.distance_func(test_user_copy, recommender_with_inf.users[id])) 
         for id, _ in neighbors_with_inf],
        users_dict,
        "Exercise 10.2 - Recommendations With Influencer Weighting"
    )
    
    logger.info("\nConclusion for Exercise 10.2:")
    logger.info("Influencer weighting gives more importance to the ratings from designated")
    logger.info("influencers. This can be implemented by treating each influencer's vote as")
    logger.info("multiple votes or by explicitly increasing their weight in the weighted average.")
    logger.info("The effect is that recommendations become more biased toward the tastes of influencers.")

# -------------------- Exercise 10.3 - Optimal K Value --------------------

def exercise_10_3():
    """
    Exercise 10.3: Determine the optimal K value for Netflix's recommendation system.
    """
    logger.info("\nExercise 10.3: Optimal K Value")
    logger.info("-" * 60)
    
    # Create a larger set of users to simulate Netflix's system (at a much smaller scale)
    num_users = 100
    
    # Create a recommender system
    recommender = MovieRecommender(normalize_ratings=True)
    
    # List of movies
    movies = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "Pulp Fiction", "Fight Club", "Forrest Gump", "Inception",
        "The Matrix", "Goodfellas", "The Silence of the Lambs",
        "Star Wars", "The Lord of the Rings", "The Avengers",
        "Jurassic Park", "Titanic", "Avatar", "Finding Nemo",
        "The Lion King", "Toy Story", "Up", "Inside Out",
        "The Social Network", "The Departed", "Gladiator",
        "Black Panther", "Wonder Woman", "Iron Man", "Spider-Man",
        "Batman Begins", "The Incredibles"
    ]
    
    # Generate users with random ratings
    for i in range(num_users):
        user_id = f"User{i+1}"
        user = User(user_id)
        
        # Randomly select 10-20 movies to rate
        num_ratings = random.randint(10, 20)
        for _ in range(num_ratings):
            movie = random.choice(movies)
            rating = random.randint(1, 5)
            user.rate_movie(movie, rating)
        
        recommender.add_user(user)
    
    # Make 5% of users influencers
    num_influencers = num_users // 20
    influencer_ids = random.sample(list(recommender.users.keys()), num_influencers)
    for user_id in influencer_ids:
        recommender.users[user_id].is_influencer = True
        recommender.users[user_id].influence_weight = 2.0
    
    # Calculate the value of k based on sqrt(N)
    sqrt_n_k = recommender.calculate_sqrt_n_recommendation()
    logger.info(f"Number of users: {num_users}")
    logger.info(f"Recommended k based on sqrt(N): {sqrt_n_k}")
    
    # Test a range of k values
    max_k = min(20, num_users - 1)  # Don't test more than num_users-1
    optimal_k, min_error = recommender.calculate_optimal_k(max_k=max_k)
    
    logger.info(f"Optimal k based on validation: {optimal_k}")
    logger.info(f"Minimum error: {min_error:.4f}")
    
    # Collect error data for visualization
    k_values = list(range(1, max_k + 1))
    errors = []
    
    # Simple error estimation for visualization
    for k in k_values:
        # Use a simplified error estimation
        total_error = 0
        count = 0
        
        # Sample a few users for error estimation
        sample_users = random.sample(list(recommender.users.keys()), min(10, num_users))
        
        for user_id in sample_users:
            user = recommender.users[user_id]
            
            # For each user, hold out a random rating
            if len(user.ratings) > 1:
                test_movie, test_rating = random.choice(list(user.ratings.items()))
                
                # Temporarily remove this rating
                user.ratings.pop(test_movie)
                
                # Get recommendations with current k
                neighbors = recommender.find_nearest_neighbors(user_id, k=k)
                
                # Direct prediction calculation
                weighted_sum = 0
                weight_sum = 0
                
                for neighbor_id, distance in neighbors:
                    neighbor = recommender.users[neighbor_id]
                    neighbor_rating = neighbor.get_rating(test_movie)
                    
                    if neighbor_rating is not None:
                        weight = 1 / (distance + 0.1)
                        if recommender.use_influencers:
                            weight *= neighbor.influence_weight
                        
                        weighted_sum += neighbor_rating * weight
                        weight_sum += weight
                
                # Calculate error if prediction exists
                if weight_sum > 0:
                    predicted_rating = weighted_sum / weight_sum
                    total_error += (predicted_rating - test_rating) ** 2
                    count += 1
                
                # Restore the rating
                user.rate_movie(test_movie, test_rating)
        
        # Calculate mean squared error
        mse = total_error / count if count > 0 else float('inf')
        errors.append(mse)
    
    # Visualize k selection
    visualize_k_selection(k_values, errors, optimal_k, sqrt_n_k, 
                        "Exercise 10.3 - Optimal K Selection")
    
    logger.info("\nConclusion for Exercise 10.3:")
    logger.info("For a large system like Netflix with millions of users, using k=5 is too small.")
    logger.info("A common rule of thumb is to use k = sqrt(N), where N is the number of users.")
    logger.info("With millions of users, this would suggest a much larger k value.")
    logger.info("Using more neighbors helps reduce the impact of outliers and provides more robust")
    logger.info("recommendations, especially for users with unique preferences.")

# -------------------- Main Function --------------------

def main():
    """Main function to run the K-Nearest Neighbors demonstrations."""
    logger.info("=" * 80)
    logger.info("K-Nearest Neighbors and Recommendation Systems - Chapter 10 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run Exercise 10.1 - Rating Normalization
        exercise_10_1()
        
        # Run Exercise 10.2 - Influencer Weighting
        exercise_10_2()
        
        # Run Exercise 10.3 - Optimal K Value
        exercise_10_3()
        
        logger.info("\nSummary of Chapter 10 exercises:")
        logger.info("10.1 Rating Normalization - Accounts for different rating scales")
        logger.info("10.2 Influencer Weighting - Gives more weight to designated influencers")
        logger.info("10.3 Optimal K Value - Use sqrt(N) for large systems like Netflix")
        
        logger.info("\nAll exercises completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()