"""
Credibility Scoring Engine
Calculates ideal scores for products based on continuous percentile ranking.
Uses proportional scoring instead of binary thresholds for more accurate results.
"""
import pandas as pd
from typing import List, Dict, Optional


class CredibilityScorer:
    """Calculate credibility scores for electronics products using continuous scoring."""
    
    def __init__(self):
        # Scoring configuration with weights (importance)
        self.score_cols_with_weights = {
            "processor_score": 8,
            "ram_gb": 7,
            "storage_gb": 6,
            "battery_mah": 5,
            "screen_inches": 4,
            "camera_mp": 4,
            "price_usd": 3,
            "weight_g": 2,
        }
        
        # Lower is better for these columns
        self.smaller_is_better = ["price_usd", "weight_g"]
        
        # Total possible weight (for normalization)
        self.max_weight = sum(self.score_cols_with_weights.values())
    
    def calculate_scores(self, products: List[Dict]) -> pd.DataFrame:
        """
        Calculate scores for a list of products using continuous percentile ranking.
        
        Each spec contributes: weight × percentile_rank (0.0 to 1.0)
        Final score normalized to 0-100 scale.
        
        Args:
            products: List of product dicts with spec keys
            
        Returns:
            DataFrame with products and calculated scores
        """
        if not products:
            return pd.DataFrame()
        
        df = pd.DataFrame(products)
        
        # Filter to only columns we can score
        available_score_cols = {k: v for k, v in self.score_cols_with_weights.items() 
                                if k in df.columns}
        
        if not available_score_cols:
            df['ideal_score'] = None
            return df
        
        score_cols = list(available_score_cols.keys())
        
        # Convert to numeric
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        
        # Only score products that have at least SOME data
        df_to_score = df[df[score_cols].notna().any(axis=1)].copy()
        
        if len(df_to_score) == 0:
            df['ideal_score'] = None
            return df
        
        # Initialize score column
        df_to_score['ideal_score'] = 0.0
        
        # Track actual weight used (in case some columns are missing)
        df_to_score['weight_used'] = 0.0
        
        # Calculate continuous percentile scores for each column
        for col, weight in available_score_cols.items():
            # Skip if column has no valid values
            if df_to_score[col].notna().sum() == 0:
                continue
            
            # Calculate percentile rank (0.0 to 1.0)
            # For "smaller is better" columns, invert the ranking
            if col in self.smaller_is_better:
                # Lower values get higher percentile (invert)
                percentiles = 1 - df_to_score[col].rank(pct=True, na_option='keep')
            else:
                # Higher values get higher percentile
                percentiles = df_to_score[col].rank(pct=True, na_option='keep')
            
            # Add weighted percentile to score (only for non-null values)
            mask = df_to_score[col].notna()
            df_to_score.loc[mask, 'ideal_score'] += percentiles[mask] * weight
            df_to_score.loc[mask, 'weight_used'] += weight
        
        # Normalize to 0-100 scale based on weight used
        # This handles products with missing specs fairly
        df_to_score['ideal_score'] = (
            df_to_score['ideal_score'] / df_to_score['weight_used'].clip(lower=1)
        ) * 100
        
        # Round to 2 decimal places
        df_to_score['ideal_score'] = df_to_score['ideal_score'].round(2)
        
        # Clean up temp column
        df_to_score = df_to_score.drop(columns=['weight_used'])
        
        # Merge scores back to original dataframe
        df.loc[df_to_score.index, 'ideal_score'] = df_to_score['ideal_score']
        
        return df
    
    def score_new_product(self, new_product: Dict, existing_products: List[Dict]) -> float:
        """
        Score a single new product against existing products.
        
        Args:
            new_product: Dict with product specs
            existing_products: List of existing product dicts
            
        Returns:
            Calculated ideal score (0-100)
        """
        # Combine new product with existing ones
        all_products = existing_products + [new_product]
        
        # Calculate scores for all
        scored_df = self.calculate_scores(all_products)
        
        # Return score for the new product (last row)
        if len(scored_df) > 0:
            score = scored_df.iloc[-1]['ideal_score']
            return round(float(score), 2) if pd.notna(score) else None
        
        return None


def score_product_from_db(product_dict: Dict, db_session) -> Optional[float]:
    """
    Helper function to score a product using all products from database.
    
    Args:
        product_dict: New product data as dict
        db_session: SQLAlchemy session
        
    Returns:
        Calculated score or None
    """
    from app import Product
    
    # Get all existing products from DB
    existing = db_session.query(Product).all()
    
    existing_dicts = [{
        'processor_score': p.processor_score,
        'ram_gb': p.ram_gb,
        'storage_gb': p.storage_gb,
        'battery_mah': p.battery_mah,
        'screen_inches': p.screen_inches,
        'camera_mp': p.camera_mp,
        'price_usd': p.price_usd,
        'weight_g': p.weight_g,
    } for p in existing]
    
    scorer = CredibilityScorer()
    return scorer.score_new_product(product_dict, existing_dicts)


def recalculate_all_scores(db_session) -> int:
    """
    Recalculate ideal_score for ALL products in the database using continuous scoring.
    
    Args:
        db_session: SQLAlchemy session
        
    Returns:
        Number of products updated
    """
    from app import Product
    
    # Get all products
    products = db_session.query(Product).all()
    
    if not products:
        return 0
    
    # Convert to dicts for scoring
    product_dicts = [{
        'id': p.id,
        'processor_score': p.processor_score,
        'ram_gb': p.ram_gb,
        'storage_gb': p.storage_gb,
        'battery_mah': p.battery_mah,
        'screen_inches': p.screen_inches,
        'camera_mp': p.camera_mp,
        'price_usd': p.price_usd,
        'weight_g': p.weight_g,
    } for p in products]
    
    # Calculate new scores
    scorer = CredibilityScorer()
    scored_df = scorer.calculate_scores(product_dicts)
    
    # Update products in database
    updated_count = 0
    for _, row in scored_df.iterrows():
        product = db_session.query(Product).get(row['id'])
        if product and pd.notna(row['ideal_score']):
            product.ideal_score = row['ideal_score']
            updated_count += 1
    
    db_session.commit()
    return updated_count
